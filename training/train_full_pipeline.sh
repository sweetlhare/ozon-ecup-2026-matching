#!/usr/bin/env bash
set -euo pipefail

# Selected final lineage after the two foundation cross-encoders. The
# foundations themselves are built by training/train_foundations.sh.

PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONPATH="data_pipeline:training${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE_FOUNDATION=${GATE_FOUNDATION:-artifacts/ce_mbert_pair_cat_refresh_w30}
SCORER_FOUNDATION=${SCORER_FOUNDATION:-artifacts/ce_mmbert_quartet_cat_refresh_w30_order}

require_model() {
  if [[ ! -s "$1/model.safetensors" ]]; then
    echo "Missing checkpoint: $1/model.safetensors" >&2
    exit 2
  fi
}

require_model "$GATE_FOUNDATION"
require_model "$SCORER_FOUNDATION"

# 1. Human-label refresh of the compact gate: exactly 750 optimizer steps.
"$PYTHON" -u training/train_ce.py \
  --model "$GATE_FOUNDATION" \
  --tag gate_raw750 \
  --max_len 384 --swap_aug --cat_batch \
  --batch 64 --eval_batch 64 --lr 2e-6 --epochs 0.1646 \
  --workers 4 --ft_eval_every 750 --judge "" --seed 42 \
  --attn_implementation sdpa

# 2. Gate specialization on the large high-purity closure pool.
"$PYTHON" -u training/train_ce.py \
  --model artifacts/ce_gate_raw750 \
  --pretrain data/closure_big \
  --tag gate_closure_big_s408 \
  --pretrain_steps 408 --pretrain_lr 2e-6 --pretrain_only \
  --ckpt_every 408 --ckpt_dir artifacts/gate_closure_big_s408_ckpt \
  --eval_every 408 --max_len 384 --order --swap_aug \
  --batch 50 --eval_batch 64 --workers 4 --judge "" --seed 42 \
  --attn_implementation sdpa

# 3. Dual-head scorer.  The production head learns organizer k=9 closure;
# the auxiliary head learns the human boundary.  The historical joint loader
# did not swap pairs, so --swap_aug is intentionally absent here.
"$PYTHON" -u training/train_ce.py \
  --model "$SCORER_FOUNDATION" \
  --pretrain data/closure_all20 \
  --tag dualhead_closure408 \
  --joint --dual_head --joint_steps 408 \
  --gold_frac 0.20 --weak_w 1.0 --llm_binarize 1.0 \
  --max_len 384 --order --batch 50 --eval_batch 64 --workers 4 \
  --lr 2e-6 --judge "" --seed 42 --attn_implementation sdpa

# 4. Preserve both FP32 heads, export the deployable FP16 production model,
# and move its classifier 10% toward the auxiliary human head.
"$PYTHON" training/export_dual_head.py \
  --source artifacts/ce_dualhead_closure408 \
  --model-out artifacts/ce_dualhead_closure408_fp16 \
  --heads-out artifacts/dualhead408_heads.safetensors
"$PYTHON" training/merge_dual_heads.py \
  --model artifacts/ce_dualhead_closure408_fp16/model.safetensors \
  --heads artifacts/dualhead408_heads.safetensors \
  --alpha 0.10 \
  --out artifacts/dualhead408_blend_a10.safetensors
cp -R artifacts/ce_dualhead_closure408_fp16 artifacts/safe_scorer
cp artifacts/dualhead408_blend_a10.safetensors artifacts/safe_scorer/model.safetensors

# 5. The main solution branches from the selected safe parents.
SOFT=1 SRC=artifacts/ce_gate_closure_big_s408_pretrained \
OUT=artifacts/sgate_670 POOL=data/soft_train.parquet \
STEPS=670 LR=2e-6 BS=50 MAXLEN=640 SEED=7 \
"$PYTHON" -u training/dose_scorer.py

SOFT=1 SRC=artifacts/ce_dualhead_closure408_fp16 \
OUT=artifacts/mix408 POOL=data/mix_train.parquet \
STEPS=408 LR=2e-6 BS=50 MAXLEN=640 SEED=7 \
"$PYTHON" -u training/dose_scorer.py

echo "Built safe gate/scorer and primary gate/scorer under artifacts/."
