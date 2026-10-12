#!/usr/bin/env bash
# Reproduce every number and figure in the project, in order.
# Total on an RTX 3050 laptop GPU: ~2.5 hours (most of it is the 3-seed ablation study).
#
#   bash scripts/reproduce.sh            # everything
#   SKIP_ABLATION=1 bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY scripts/prepare_data.py
$PY -m pytest -q

# architecture evidence (Figures 1-5)
$PY -m experiments.hand_attention
$PY -m experiments.mask_visualization
$PY -m experiments.shape_trace
$PY -m experiments.positional_encoding_viz
$PY -m experiments.model_summary --config configs/tiny_gpt.yaml

# debugging gate, LR sweep, main training
$PY -m src.train --config configs/overfit.yaml
$PY -m experiments.tune --steps 1500 --lrs 3e-4 6e-4 1e-3 2e-3 3e-3 4e-3 --dropouts 0.2
$PY -m src.train --config configs/tiny_gpt.yaml
$PY -c "from src.checkpoint import export_inference_checkpoint as e; e('checkpoints/tiny_gpt/best.pt', 'checkpoints/minigpt_shakespeare.pt')"

# evaluation of the trained model (Figures 6-8)
CKPT=checkpoints/minigpt_shakespeare.pt
$PY -m experiments.baselines
$PY -m experiments.plot_training --run outputs/logs/tiny_gpt
$PY -m experiments.evaluate --checkpoint $CKPT
$PY -m experiments.attention_visualization --checkpoint $CKPT
$PY -m experiments.model_summary --checkpoint $CKPT
$PY -m experiments.decoding_compare --checkpoint $CKPT

# concept experiments
$PY -m experiments.rnn_vs_transformer
$PY -m experiments.lstm_baseline
$PY -m experiments.mlm_vs_clm --steps 1500

# controlled ablations (Figure 9)
if [ -z "${SKIP_ABLATION:-}" ]; then
  $PY -m experiments.ablation --seeds 1337 1338 1339
fi

$PY scripts/acceptance.py --checkpoint $CKPT
