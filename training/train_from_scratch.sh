#!/usr/bin/env bash
set -euo pipefail

# One entry point: organizer parquet files + public Hugging Face checkpoints
# -> every intermediate dataset -> four final checkpoints -> two submission ZIPs.

PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

"$PYTHON" scripts/build_training_data.py "$@"
"$PYTHON" training/download_base_models.py
PYTHON="$PYTHON" bash training/train_foundations.sh

GATE_FOUNDATION=artifacts/ce_mbert_pair_cat_refresh_w30 \
SCORER_FOUNDATION=artifacts/ce_mmbert_quartet_cat_refresh_w30_order \
PYTHON="$PYTHON" \
bash training/train_full_pipeline.sh
bash training/install_final_models.sh

"$PYTHON" scripts/submission.py build mix408_textneg --force --trained \
  --output submissions/trained_mix408_textneg.zip
"$PYTHON" scripts/submission.py build final_safe_fwdgate --force --trained \
  --output submissions/trained_final_safe_fwdgate.zip
