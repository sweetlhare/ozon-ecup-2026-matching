#!/usr/bin/env bash
set -euo pipefail

# Exact last-stage recipes. Run from the repository root after building data.
# Parents can come from train_full_pipeline.sh or an extracted checkpoint bundle.

export PYTHONPATH="${PYTHONPATH:-}:data_pipeline"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE_PARENT=${GATE_PARENT:-models/final_safe_fwdgate/gate}
SCORER_PARENT=${SCORER_PARENT:-artifacts/ce_dualhead_closure408_fp16}
for parent in "$GATE_PARENT" "$SCORER_PARENT"; do
  if [[ ! -s "$parent/model.safetensors" ]]; then
    echo "Missing parent checkpoint: $parent/model.safetensors" >&2
    exit 2
  fi
done

SOFT=1 \
SRC="$GATE_PARENT" \
OUT=artifacts/sgate_670 \
POOL=data/soft_train.parquet \
STEPS=670 LR=2e-6 BS=50 MAXLEN=640 SEED=7 \
python -u training/dose_scorer.py

SOFT=1 \
SRC="$SCORER_PARENT" \
OUT=artifacts/mix408 \
POOL=data/mix_train.parquet \
STEPS=408 LR=2e-6 BS=50 MAXLEN=640 SEED=7 \
python -u training/dose_scorer.py

echo "Training complete. Compare the resulting model.safetensors files with:"
echo "  primary gate   ab034b31201d632d494360a7ea2a6c56f878ef507fbcb9b7a23ec325d6200c24"
echo "  primary scorer baeeb64e6ba3f20562b1837a4bff590d14148b3b3a394fa46c3b1799619c33e0"
