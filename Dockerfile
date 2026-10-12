FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src/ src/

# NumPy weights (scripts/export_numpy.py) are fetched from the GitHub Release at build time (not stored in git).
ARG MINIGPT_CHECKPOINT_URL=https://github.com/AbdulRahmanAzam/minigpt-shakespeare/releases/download/v1.0/minigpt_shakespeare.npz
ENV MINIGPT_CHECKPOINT=/app/checkpoints/minigpt_shakespeare.npz
RUN python -c "import sys, urllib.request, pathlib; pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "${MINIGPT_CHECKPOINT_URL}" "${MINIGPT_CHECKPOINT}" \
 && echo "57c3bfa1903578550c96b001ec18c4fb4ac0bfdc00c3b0b8fb0628d4ed266356  ${MINIGPT_CHECKPOINT}" | sha256sum -c -

EXPOSE 8000
# Render (and most PaaS) inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
