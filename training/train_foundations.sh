#!/usr/bin/env bash
set -euo pipefail

# Rebuild both foundation cross-encoders from public Hugging Face checkpoints
# and organizer data. Run from the repository root on one CUDA GPU.

PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONPATH="data_pipeline:training${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUMODERNBERT=${RUMODERNBERT:-models/base/rumodernbert}
MMBERT=${MMBERT:-models/base/mmbert}
GRANITE=${GRANITE:-models/base/granite311}

for file in \
  data/matches.parquet data/items_human.parquet data/val_maxsim.npy \
  data/llmcand_pairs.parquet data/llmcand_items.parquet \
  data/llmcand2_pairs.parquet data/llmcand2_items.parquet \
  data/llmfull_ord_pairs.parquet data/llmfull_ord_items.parquet; do
  if [[ ! -s "$file" ]]; then
    echo "Missing training input: $file" >&2
    exit 2
  fi
done

train() {
  "$PYTHON" -u training/train_ce.py "$@" \
    --judge "" --seed 42
}

# ---------------------------------------------------------------------------
# Gate lineage: RuModernBERT scale teachers -> pair-margin student -> refresh.
# ---------------------------------------------------------------------------

# Original 60k retrieval-domain teacher.
train --model "$RUMODERNBERT" --tag mbert_cand \
  --pretrain data/llmcand --pretrain_steps 60000 --pretrain_lr 5e-5 \
  --save_after_pretrain --max_len 384 --swap_aug \
  --batch 32 --eval_batch 64 --lr 2e-5 --epochs 2 \
  --workers 6 --eval_every 20000 --ckpt_every 20000 \
  --ckpt_dir artifacts/mbert_cand_ckpt \
  --phase_best_dir artifacts/mbert_cand_recovery

# 220k scale teacher and the common student initialization.
train --model "$RUMODERNBERT" --tag mbert_x4 \
  --pretrain data/llmcand2 --pretrain_steps 220000 --pretrain_lr 5e-5 \
  --save_after_pretrain --max_len 384 --swap_aug \
  --batch 32 --eval_batch 64 --lr 2e-5 --epochs 2 \
  --workers 6 --eval_every 20000 --ckpt_every 20000 \
  --ckpt_dir artifacts/mbert_x4_ckpt \
  --phase_best_dir artifacts/mbert_x4_recovery

"$PYTHON" -u training/distill_score.py \
  --data data --out data/distill_mbert_cand_x4_train.parquet \
  --models artifacts/ce_mbert_cand artifacts/ce_mbert_x4 \
  --max-len 384 --batch 64 --workers 4

train --model artifacts/ce_mbert_x4_pretrained --tag mbert_pair_distill_w03 \
  --max_len 384 --swap_aug --batch 16 --accum 2 --eval_batch 64 \
  --lr 2e-5 --epochs 2 --workers 4 \
  --teacher data/distill_mbert_cand_x4_train.parquet --distill_w 0.3 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mbert_pair_distill_w03_recovery

train --model artifacts/ce_mbert_pair_distill_w03 --tag mbert_pair_refresh_w10 \
  --max_len 384 --swap_aug --batch 32 --eval_batch 64 \
  --lr 5e-6 --epochs 1 --workers 4 \
  --teacher data/distill_mbert_cand_x4_train.parquet --distill_w 1.0 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mbert_pair_refresh_w10_recovery

train --model artifacts/ce_mbert_pair_refresh_w10 \
  --tag mbert_pair_cat_refresh_w30 \
  --max_len 384 --swap_aug --cat_batch --batch 32 --eval_batch 64 \
  --lr 5e-6 --epochs 1 --workers 4 \
  --teacher data/distill_mbert_cand_x4_train.parquet --distill_w 3.0 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mbert_pair_cat_refresh_w30_recovery

# ---------------------------------------------------------------------------
# Scorer lineage: mmBERT scale teachers + ordered ensemble distillation.
# ---------------------------------------------------------------------------

# Shared 220k domain checkpoint and the ordinary gold teacher. Category weights
# compensate the retrieval pool's category skew for the macro metric.
train --model "$MMBERT" --tag mmbert_x4b \
  --pretrain data/llmcand2 --pretrain_steps 220000 --pretrain_lr 5e-5 \
  --save_after_pretrain --max_len 384 --swap_aug --cat_weight 1.0 --len_bucket \
  --batch 24 --eval_batch 48 --lr 2e-5 --epochs 2 \
  --workers 4 --eval_every 20000 --ckpt_every 20000 \
  --ckpt_dir artifacts/mmbert_x4b_ckpt \
  --phase_best_dir artifacts/mmbert_x4b_recovery

# Same encoder checkpoint, different deterministic attribute order.
train --model artifacts/ce_mmbert_x4b_pretrained --tag mmbert_x4b_order \
  --max_len 384 --swap_aug --order --batch 24 --eval_batch 48 \
  --lr 2e-5 --epochs 2 --workers 3 --ft_eval_every 1500 \
  --phase_best_dir artifacts/mmbert_x4b_order_recovery

# Independent public architecture used as one teacher voice.
train --model "$GRANITE" --tag granite311_order \
  --max_len 384 --swap_aug --order --len_bucket \
  --batch 32 --eval_batch 64 --lr 2e-5 --epochs 2 --workers 4 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/granite311_order_recovery

# Full-pool ordered mmBERT voice. The selected historical branch used the
# checkpoint at 100k, followed by an ordinary two-epoch human fine-tune.
train --model "$MMBERT" --tag mmbert_combo100k \
  --pretrain data/llmfull_ord --pretrain_steps 100000 --pretrain_lr 5e-5 \
  --pretrain_only --save_after_pretrain --max_len 384 --swap_aug --order \
  --len_bucket --batch 32 --eval_batch 64 --workers 4 \
  --eval_every 20000 --ckpt_every 20000 \
  --ckpt_dir artifacts/mmbert_combo100k_ckpt

train --model artifacts/ce_mmbert_combo100k_pretrained \
  --tag mmbert_combo100k_gold \
  --max_len 384 --swap_aug --order --len_bucket \
  --batch 32 --eval_batch 64 --lr 2e-5 --epochs 2 --workers 4 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mmbert_combo100k_gold_recovery

"$PYTHON" -u training/distill_score.py \
  --data data --out data/distill_x4b_x4bo_granite_combo_train.parquet \
  --models artifacts/ce_mmbert_x4b artifacts/ce_mmbert_x4b_order \
           artifacts/ce_granite311_order artifacts/ce_mmbert_combo100k_gold \
  --max-len 384 --batch 64 --workers 4

train --model artifacts/ce_mmbert_x4b_pretrained \
  --tag mmbert_distill_x4b_x4bo_granite_combo_w10_order \
  --max_len 384 --swap_aug --order --batch 24 --eval_batch 48 \
  --lr 2e-5 --epochs 2 --workers 4 \
  --teacher data/distill_x4b_x4bo_granite_combo_train.parquet --distill_w 1.0 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mmbert_distill_combo_recovery

train --model artifacts/ce_mmbert_distill_x4b_x4bo_granite_combo_w10_order \
  --tag mmbert_quartet_cat_refresh_w30_order \
  --max_len 384 --swap_aug --order --cat_batch --batch 24 --eval_batch 48 \
  --lr 5e-6 --epochs 1 --workers 4 \
  --teacher data/distill_x4b_x4bo_granite_combo_train.parquet --distill_w 3.0 \
  --ft_eval_every 1500 \
  --phase_best_dir artifacts/mmbert_quartet_cat_refresh_w30_order_recovery

echo "Foundations built:"
echo "  artifacts/ce_mbert_pair_cat_refresh_w30"
echo "  artifacts/ce_mmbert_quartet_cat_refresh_w30_order"
