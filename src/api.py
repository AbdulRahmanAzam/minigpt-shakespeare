"""REST API for MiniGPT (FastAPI).

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Environment variables:
    MINIGPT_CHECKPOINT      path to the checkpoint (default: checkpoints/minigpt_shakespeare.pt)
    MINIGPT_CHECKPOINT_URL  if the file is missing, download it from here at startup (used on Render)
    MINIGPT_MAX_NEW_TOKENS  upper limit per request (default 1000)
    MINIGPT_API_KEYS        comma-separated keys; if set, POST /generate needs an "X-API-Key" header

Endpoints:
    GET  /            small HTML page for trying the model in a browser
    GET  /health      liveness check
    GET  /model-info  architecture, parameter count, training metrics
    POST /generate    {"prompt": "ROMEO:", "max_new_tokens": 200, "temperature": 0.8, ...}
"""

from __future__ import annotations

import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.checkpoint import load_checkpoint
from src.generate import generate_text, resolve_checkpoint
from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer

MAX_NEW_TOKENS = int(os.environ.get("MINIGPT_MAX_NEW_TOKENS", "1000"))


def keys_from_env() -> list[str]:
    return [k.strip() for k in os.environ.get("MINIGPT_API_KEYS", "").split(",") if k.strip()]


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=2000, description="text to continue")
    max_new_tokens: int = Field(200, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy")
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int | None = Field(None, description="fix for reproducible output")


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    text: str
    ignored_chars: list[str]
    new_tokens: int
    seconds: float
    temperature: float
    seed: int | None


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def load_default_model() -> tuple[MiniGPT, CharTokenizer, dict[str, Any]]:
    path = resolve_checkpoint(os.environ.get("MINIGPT_CHECKPOINT"))
    url = os.environ.get("MINIGPT_CHECKPOINT_URL")
    if not path.exists() and url:
        _download(url, path)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return load_checkpoint(path, "cpu")


def create_app(model: MiniGPT | None = None, tokenizer: CharTokenizer | None = None,
               state: dict[str, Any] | None = None, api_keys: list[str] | None = None) -> FastAPI:
    """Build the app. Tests inject a small model; in production the checkpoint is loaded at startup.

    With no API keys configured, /generate is open (local use); otherwise it requires a valid X-API-Key.
    """
    keys = keys_from_env() if api_keys is None else api_keys
    holder: dict[str, Any] = {"model": model, "tokenizer": tokenizer, "state": state or {}}
    lock = threading.Lock()   # one generation at a time keeps CPU usage predictable on small hosts

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if holder["model"] is None:
            holder["model"], holder["tokenizer"], holder["state"] = load_default_model()
        yield

    def require_key(x_api_key: str | None = Header(None)) -> None:
        if keys and not any(secrets.compare_digest(x_api_key or "", k) for k in keys):
            raise HTTPException(401, "missing or invalid X-API-Key header")

    app = FastAPI(title="MiniGPT Shakespeare", version="1.0.0", lifespan=lifespan,
                  description="Character-level GPT implemented from scratch in PyTorch.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": holder["model"] is not None}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        m: MiniGPT = holder["model"]
        st = holder["state"]
        return {
            "parameters": m.num_parameters(),
            "config": m.config_dict(),
            "vocab": "".join(holder["tokenizer"].chars),
            "training_step": st.get("step"),
            "metrics": st.get("metrics", {}),
            "max_new_tokens_limit": MAX_NEW_TOKENS,
        }

    @app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_key)])
    def generate(req: GenerateRequest) -> GenerateResponse:
        tok: CharTokenizer = holder["tokenizer"]
        if req.prompt and not tok.encode(req.prompt, strict=False):
            raise HTTPException(422, f"prompt contains no characters from the model vocabulary: {tok.chars!r}")
        with lock:
            r = generate_text(holder["model"], tok, req.prompt, req.max_new_tokens, req.temperature,
                              req.top_k, req.top_p, req.seed)
        return GenerateResponse(prompt=r.prompt, completion=r.completion, text=r.text,
                                ignored_chars=r.ignored_chars, new_tokens=r.new_tokens,
                                seconds=round(r.seconds, 3), temperature=req.temperature, seed=req.seed)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    return app


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniGPT Shakespeare</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1d2433}
 textarea,input{font:inherit;padding:.4rem;border:1px solid #c5cbd6;border-radius:6px}
 textarea{width:100%;height:4.5rem;box-sizing:border-box}
 .row{display:flex;gap:1rem;flex-wrap:wrap;margin:.6rem 0;align-items:center}
 label{font-size:.9rem} input{width:6rem}
 button{font:inherit;padding:.5rem 1.2rem;border:0;border-radius:6px;background:#2856c7;color:#fff;cursor:pointer}
 button:disabled{opacity:.6}
 pre{white-space:pre-wrap;background:#f4f6fa;padding:1rem;border-radius:8px;min-height:6rem}
 small{color:#5b6475}
</style></head><body>
<h1>MiniGPT &mdash; Tiny Shakespeare</h1>
<p><small id="info">loading model info&hellip;</small></p>
<textarea id="prompt">ROMEO:</textarea>
<div class="row">
 <label>max new tokens <input id="n" type="number" value="300" min="1" max="1000"></label>
 <label>temperature <input id="t" type="number" value="0.8" step="0.1" min="0" max="2"></label>
 <label>top-k <input id="k" type="number" placeholder="off" min="1"></label>
 <label>seed <input id="s" type="number" placeholder="random"></label>
 <label>API key <input id="key" type="password" style="width:12rem"></label>
 <button id="go">Generate</button>
</div>
<pre id="out"></pre>
<p><small>API: <code>POST /generate</code> &middot; <a href="/docs">interactive docs</a> &middot;
<a href="/model-info">model info</a></small></p>
<script>
const $ = id => document.getElementById(id);
$('key').value = localStorage.getItem('minigpt_key') || '';
fetch('/model-info').then(r => r.json()).then(m => {
  const c = m.config;
  $('info').textContent = `${(m.parameters/1e6).toFixed(2)}M parameters | ${c.n_layers} layers, ${c.n_heads} heads, ` +
    `d_model ${c.d_model}, context ${c.context_length}, character vocab ${c.vocab_size}` +
    (m.metrics.val_loss ? ` | val loss ${m.metrics.val_loss.toFixed(3)}` : '');
});
$('go').onclick = async () => {
  const num = id => $(id).value === '' ? null : Number($(id).value);
  $('go').disabled = true; $('out').textContent = 'generating...';
  try {
    localStorage.setItem('minigpt_key', $('key').value);
    const r = await fetch('/generate', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Key': $('key').value},
      body: JSON.stringify({prompt: $('prompt').value, max_new_tokens: num('n'), temperature: num('t'),
                            top_k: num('k'), seed: num('s')})});
    const j = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(j.detail));
    $('out').textContent = j.text + `\\n\\n[${j.new_tokens} tokens in ${j.seconds}s]` +
      (j.ignored_chars.length ? `\\n[dropped characters not in vocabulary: ${j.ignored_chars.join(' ')}]` : '');
  } catch (e) { $('out').textContent = 'Error: ' + e.message; }
  $('go').disabled = false;
};
</script></body></html>"""

app = create_app()
