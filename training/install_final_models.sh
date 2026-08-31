#!/usr/bin/env bash
set -euo pipefail

# Put the four freshly trained checkpoints where the deterministic submission
# builder expects them. Runtime marker files are supplied by solutions/*.

copy_model() {
  local source=$1 destination=$2
  mkdir -p "$destination"
  for file in model.safetensors config.json tokenizer.json tokenizer_config.json; do
    if [[ ! -s "$source/$file" ]]; then
      echo "Missing trained model file: $source/$file" >&2
      exit 2
    fi
    cp "$source/$file" "$destination/$file"
  done
}

copy_model artifacts/sgate_670 models/mix408_textneg/gate
copy_model artifacts/mix408 models/mix408_textneg/scorer
copy_model artifacts/ce_gate_closure_big_s408_pretrained \
  models/final_safe_fwdgate/gate
copy_model artifacts/safe_scorer models/final_safe_fwdgate/scorer

echo "Installed four trained checkpoints under models/."
