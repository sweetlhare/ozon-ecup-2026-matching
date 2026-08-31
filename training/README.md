# Обучение

## Компоненты

- `train_ce.py` — основной cross-encoder trainer: pretraining, fine-tuning,
  component-level split, category-aware sampling и дополнительные ranking
  losses.
- `dose_scorer.py` — короткое финальное дообучение готового checkpoint.
- `split.py` — разделение по компонентам связности.
- `metrics.py` — Macro PR-AUC через
  `sklearn.metrics.average_precision_score`.
- `export_fp16.py` — экспорт checkpoint в FP16 для submission.
- `train_full_pipeline.sh` — выбранная цепочка от foundation-checkpoint до
  четырёх весов двух финальных решений.
- `train_foundations.sh` — обучение обеих foundation-моделей из публичных
  pretrained-моделей и файлов организаторов.
- `train_from_scratch.sh` — единый запуск всей цепочки и сборка двух ZIP.
- `install_final_models.sh` — раскладывает четыре обученных checkpoint по
  каталогам, из которых детерминированно собираются два submission.
- `base_models.json` и `download_base_models.py` — фиксируют immutable revision
  трёх публичных pretrained-моделей и скачивают их перед обучением.
- `train_final_models.sh` — только два последних запуска, если safe parent уже
  скачаны.
- `export_dual_head.py` и `merge_dual_heads.py` — сохранение двух голов и
  A10-интерполяция, использованная safe scorer.

## Полная выбранная цепочка

```text
deepvk/RuModernBERT-base
  ├─ llmcand, 60k ──────────── teacher A
  └─ llmcand2, 220k ────────── teacher B + student initialization
       └─ teacher margin distillation (A+B), w=0.3
           └─ refresh w=1.0
               └─ category refresh w=3.0 ── foundation gate

foundation gate
  └─ human refresh, 750 шагов
      └─ closure_big, 408 шагов ───────────── safe gate
                                             └─ soft_train, 670 ─ primary gate

jhu-clsp/mmBERT-base
  ├─ llmcand2, 220k ────────── ordinary + ORDERED teacher
  └─ llmfull_ord, 100k ─────── full-pool ORDERED teacher
ibm-granite/granite-embedding-311m-multilingual-r2
  └─ human ORDERED ─────────── independent teacher
four-teacher margin distillation
  └─ category refresh w=3.0 ── foundation scorer

foundation scorer
  └─ closure_all20 + human gold, dual-head joint, 408 шагов
      ├─ production head FP16 ─────────────── mix_train, 408 ─ primary scorer
      └─ 0.90 production + 0.10 human ────── safe scorer
```

`closure408 parent` не является внешним или вручную подготовленным файлом. Его
создают шаги dual-head в `train_full_pipeline.sh` из
`closure_all20_{pairs,items}.parquet`, которые, в свою очередь, строятся
`scripts/build_training_data.py` из parquet организаторов.

Полный запуск:

```bash
bash training/train_from_scratch.sh
```

Он выполняет все стадии последовательно. Никакие вручную подготовленные
parquet, teacher-логиты или промежуточные checkpoint не требуются.
Свежие результаты получают имена `trained_mix408_textneg.zip` и
`trained_final_safe_fwdgate.zip`, чтобы не перезаписать архивы, фактически
отправленные в соревнование.

Если foundation уже обучены, вторую половину можно запустить отдельно:

```bash
bash training/train_full_pipeline.sh
```

Контрольные SHA исторических foundation-весов приведены для сверки результата:

| checkpoint | SHA-256 `model.safetensors` |
|---|---|
| gate foundation (`ce_mbert_pair_cat_refresh_w30`) | `e1c1047e209e86d0505cec56fc8a36ec6ecdc8ddc5f2cad6d4e0d1a6efb61318` |
| scorer foundation (`ce_mmbert_quartet_cat_refresh_w30_order`) | `e803379e98341d1e65eda9b5c604df8dfcf93355c458c69021dcde18128de79e` |

Все пары, gold-метки, closure-связи и teacher-корпусы строятся из файлов
организаторов. Внешними входами являются только три публичные pretrained-модели.
Для быстрой сборки готовых ZIP обучение не требуется — финальные веса находятся
в Release.

## Финальные параметры

| модель | parent | данные | шаги | batch | lr | max length | seed |
|---|---|---|---:|---:|---:|---:|---:|
| primary gate | safe gate | `soft_train.parquet` | 670 | 50 | `2e-6` | 640 | 7 |
| primary scorer | dual-head production head | `mix_train.parquet` | 408 | 50 | `2e-6` | 640 | 7 |

Обе модели обучаются с soft target `k/9`. Для scorer смесь также содержит
transitive hard negatives с target 0 и matched positives с target 1.

Запуск:

```bash
PYTHONPATH=data_pipeline bash training/train_final_models.sh
```

## Checkpoint

Все четыре checkpoint, используемые двумя финальными решениями, находятся в
GitHub Release и скачиваются командой:

```bash
.venv/bin/python scripts/download_artifacts.py --only models
```

Контрольные SHA-256 финальных весов:

| модель | SHA-256 |
|---|---|
| primary gate | `ab034b31201d632d494360a7ea2a6c56f878ef507fbcb9b7a23ec325d6200c24` |
| primary scorer | `baeeb64e6ba3f20562b1837a4bff590d14148b3b3a394fa46c3b1799619c33e0` |
| safe gate | `34d69b25a1fa599bc9d6c9fc6240a5270128aaefe0f1c566eecdbb54f81a672c` |
| safe scorer | `6155f6fd080a4224655bf72966da4743196f16a789e8737f00147ca53ffaada9` |

Промежуточный production-head FP16 имеет SHA-256
`8cc805f8c6e71b308901049ac6c890fa0b6239ce01e75a9bb2f33ad4cb628ac5`.
Теперь код его получения и последующего A10 merge находится в репозитории.
