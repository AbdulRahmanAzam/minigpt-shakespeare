"""REST API contract, using a small randomly initialised model (no checkpoint needed)."""

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.tokenizer import CharTokenizer


@pytest.fixture
def client(tiny_model):
    chars = list("\n !',.:;?abcdefgh")[: tiny_model.cfg.vocab_size]
    tok = CharTokenizer(chars)
    app = create_app(tiny_model, tok, {"step": 5, "metrics": {"val_loss": 1.5}})
    with TestClient(app) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["parameters"] > 0 and info["config"]["n_layers"] == 2 and info["training_step"] == 5


def test_generate_returns_prompt_plus_completion(client):
    r = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 15, "temperature": 0.8, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("abc") and len(body["completion"]) == 15
    again = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 15, "temperature": 0.8, "seed": 1})
    assert again.json()["text"] == body["text"]               # seeded -> reproducible


def test_unknown_characters_are_reported(client):
    body = client.post("/generate", json={"prompt": "ab#c", "max_new_tokens": 3}).json()
    assert body["ignored_chars"] == ["#"] and body["prompt"] == "abc"


@pytest.mark.parametrize("payload", [
    {"max_new_tokens": 0}, {"max_new_tokens": 100000}, {"temperature": -1}, {"top_p": 1.5},
    {"prompt": "###"},
])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/generate", json=payload).status_code == 422


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "MiniGPT" in r.text
