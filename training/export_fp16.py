"""Атомарный экспорт обученной HF-модели в компактный FP16-каталог.

Модели train_ce сохраняются в FP32. Для инференса run.py всё равно загружает их
в bf16, поэтому FP32 в ZIP только удваивает размер. Скрипт переносит tokenizer
и служебные маркеры train-time подачи; готовый каталог появляется лишь после
успешного сохранения и повторной загрузки.
"""
import argparse
import os
import shutil
import tempfile

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MARKERS = ("ORDERED", "ALIGNED", "CATS", "SPECIALIST", "TRAIN_WEIGHTS")


def floating_dtypes(model):
    return sorted({str(t.dtype) for t in model.state_dict().values()
                   if torch.is_floating_point(t)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isdir(src):
        ap.error(f"нет каталога модели: {src}")
    if os.path.exists(dst):
        ap.error(f"выход уже существует, не перезаписываю: {dst}")
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(dst)}.tmp-", dir=parent)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            src, dtype=torch.float16, trust_remote_code=True)
        model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        tokenizer.save_pretrained(tmp)
        for marker in MARKERS:
            path = os.path.join(src, marker)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(tmp, marker))

        weights = os.path.join(tmp, "model.safetensors")
        if not os.path.isfile(weights) or os.path.getsize(weights) == 0:
            raise RuntimeError("save_pretrained не создал model.safetensors")

        # Повторная загрузка ловит неполный config/веса до атомарной публикации.
        check = AutoModelForSequenceClassification.from_pretrained(
            tmp, dtype=torch.float16, trust_remote_code=True)
        dtypes = floating_dtypes(check)
        if any(dtype not in ("torch.float16",) for dtype in dtypes):
            raise RuntimeError(f"после экспорта остались floating dtype: {dtypes}")
        del check, model

        os.replace(tmp, dst)
        print(f"OK: {src} -> {dst}")
        print(f"  model.safetensors: {os.path.getsize(os.path.join(dst, 'model.safetensors'))} bytes")
        print(f"  floating dtype: {', '.join(dtypes)}")
        present = [name for name in MARKERS if os.path.exists(os.path.join(dst, name))]
        print(f"  markers: {', '.join(present) if present else 'none'}")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

