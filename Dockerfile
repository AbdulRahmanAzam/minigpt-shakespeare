FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Weights are fetched from the GitHub Release at build time (not stored in git).
ARG MINIGPT_CHECKPOINT_URL=https://github.com/AbdulRahmanAzam/minigpt-shakespeare/releases/download/v1.0/minigpt_shakespeare.pt
ENV MINIGPT_CHECKPOINT_URL=${MINIGPT_CHECKPOINT_URL} \
    MINIGPT_CHECKPOINT=/app/checkpoints/minigpt_shakespeare.pt
RUN python scripts/download_checkpoint.py --url "${MINIGPT_CHECKPOINT_URL}" --out "${MINIGPT_CHECKPOINT}"

EXPOSE 8000
# Render (and most PaaS) inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
