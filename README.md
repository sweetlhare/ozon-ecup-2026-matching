# Ozon E-CUP 2026 — Product Matching, 3 место

Решение строит вероятность совпадения двух товарных карточек. В репозитории
зафиксированы два финальных варианта:

| решение | назначение | public Macro PR-AUC |
|---|---|---:|
| `mix408_textneg` | основной сабмит | **0.5480056269** |
| `final_safe_fwdgate` | вариант с меньшим временем инференса | **0.5426350252** |

Метрика соревнования — среднее значение `average_precision_score` по 20
категориям. Поэтому модели и финальное объединение оптимизируют порядок пар
внутри каждой категории.

## Быстрый запуск

Нужны Python 3.12, около 25 ГБ свободного места для исходных и промежуточных
данных и CUDA-совместимая видеокарта для обучения и инференса.

```bash
git clone https://github.com/sweetlhare/ozon-ecup-2026-matching.git
cd ozon-ecup-2026-matching

make setup
make data
make artifacts
make submissions
make verify
```

`make data` скачивает официальные parquet-файлы и строит обучающие выборки.
`make artifacts` скачивает четыре финальных checkpoint и два финальных ZIP из
GitHub Release. Остальные команды собирают решения и проверяют все SHA-256.

Полное переобучение, включая обе foundation-модели:

```bash
make setup
make data
make train
```

`make train` начинает с публичных `deepvk/RuModernBERT-base`,
`jhu-clsp/mmBERT-base` и
`ibm-granite/granite-embedding-311m-multilingual-r2`, строит teacher-ансамбли,
дистиллирует две foundation-модели и затем обучает четыре финальных checkpoint.
Результат упаковывается в `submissions/trained_mix408_textneg.zip` и
`submissions/trained_final_safe_fwdgate.zip`. Готовые архивы из Release
проверяются отдельно по историческим SHA; свежий GPU-прогон проверяется по
структуре, типам весов и маркерам runtime.

Если нужны только готовые решения без повторной подготовки train-данных:

```bash
make setup
make artifacts
make submissions
make verify
```

## Полный пайплайн

```text
официальные данные организатора
    items.parquet
    matches_llm.parquet
          │
          ├── component holdout + category quota + ORDERED render
          │       └── llmfull_ord (10 289 587 пар)
          │
          ├── retrieval candidate filter
          │       ├── llmcand (6 000 000 пар)
          │       └── llmcand2 (5 750 952 пары)
          │
          ├── k=9 closure + exact-input k=0 consensus из llmfull_ord
          │       ├── closure_consensus (17 категорий)
          │       ├── closure_big (gate)
          │       └── closure_all20 (dual-head scorer)
          │
          ├── component-level holdout
          │       └── llmvalS_pairs.parquet
          │
          ├── soft labels k/9, 4 000 пар на категорию
          │       └── soft_train.parquet
          │
          ├── transitive hard negatives + matched positives
          │       └── anti_train.parquet
          │
          └── soft_train + anti_train, фиксированный shuffle
                  └── mix_train.parquet
                           │
                           ├── safe gate → gate: 670 шагов
                           └── dual-head production → scorer: 408 шагов
                                    │
                                    ▼
                         два cross-encoder checkpoint
                                    │
                                    ▼
                  cascade + partial symmetry + category RRF
                                    │
                                    ▼
                           submission ZIP
```

### 1. Исходные данные

Скрипт `scripts/download_data.py` загружает официальные файлы:

- `items.parquet`;
- `items_human.parquet`;
- `matches.parquet`;
- `matches_llm.parquet`.

Для финальных derived-корпусов используются `items.parquet` и
`matches_llm.parquet`. Остальные два файла нужны базовому обучающему контуру.

### 2. Подготовка обучающих выборок

Весь этап запускается одной командой:

```bash
.venv/bin/python scripts/build_training_data.py --download
```

Сначала создаются ранние промежуточные корпуса:

| файл | строки |
|---|---:|
| `data/closure_consensus_pairs.parquet` | 20 400 |
| `data/closure_big_pairs.parquet` | 123 733 |
| `data/closure_gapcats_pairs.parquet` | 3 600 |
| `data/closure_all20_pairs.parquet` | 24 000 |

Затем создаются последние train-файлы:

| файл | строки | SHA-256 |
|---|---:|---|
| `data/soft_train.parquet` | 80 000 | `d969792a4eb7cc21b867f48d79bdffa96622a7f2d59acf7e9521774f9689bfa2` |
| `data/anti_train.parquet` | 80 000 | `a96c8a68ddbcef473d022b1c89fa0591dd1e14dc967a052aa52b366d9f91595f` |
| `data/mix_train.parquet` | 160 000 | `7a7e8c7efa54a2cfcfc16fd30babde5d5d50dc95b8a2cac6862703d39a24cd67` |

Подробности каждого шага находятся в
[`data_pipeline/README.md`](data_pipeline/README.md).

### 3. Обучение

Полный запуск записан в `training/train_from_scratch.sh`:

1. `train_foundations.sh` обучает исходные RuModernBERT/mmBERT-модели,
   формирует два teacher-корпуса на ручном train-сплите и получает gate/scorer
   foundation;
2. `train_full_pipeline.sh` выполняет raw750, `closure_big`,
   `closure_all20`, dual-head обучение, экспорт обеих голов, A10 merge и два
   финальных dose-запуска;
3. оба submission ZIP собираются и проверяются тем же сценарием.

`training/train_final_models.sh` оставлен как короткий вариант только для двух
последних запусков от уже готовых safe-parent.

Финальный этап:

- gate: `soft_train.parquet`, 670 шагов, batch 50, learning rate `2e-6`;
- scorer: `mix_train.parquet`, 408 шагов, batch 50, learning rate `2e-6`;
- `max_length=640`, seed 7, soft target `k/9`.

`training/train_ce.py` содержит базовый cross-encoder training loop,
`training/dose_scorer.py` — финальное дообучение. Полные параметры и
контрольные SHA checkpoint перечислены в
[`training/README.md`](training/README.md).

Четыре готовых финальных checkpoint опубликованы в
[Reproducibility Release](https://github.com/sweetlhare/ozon-ecup-2026-matching/releases/tag/reproducibility-v1).
Они позволяют собрать и запустить оба решения без повторного обучения.

### 4. Инференс

Каждая карточка преобразуется в текст:

```text
категория ; название ; канонизированные атрибуты в фиксированном порядке
```

Далее работает каскад:

1. Gate ранжирует все пары внутри категории.
2. Scorer пересчитывает верхние 65% кандидатов.
3. Для 75% спорных пар scorer дополнительно обрабатывает обратный порядок
   карточек.
4. Прямой и обратный проходы объединяются через Reciprocal Rank Fusion,
   `k=100`.
5. В основном решении дополнительно применяется точная коррекция известных
   text-negative signatures.

Сортировка пар по длине перед batching снижает padding и ускоряет инференс.

### 5. Сборка решений

```bash
.venv/bin/python scripts/download_artifacts.py

.venv/bin/python scripts/submission.py build mix408_textneg --force
.venv/bin/python scripts/submission.py build final_safe_fwdgate --force
```

Сборщик проверяет каждый runtime-файл, checkpoint и marker по SHA-256, создаёт
ZIP и запускает независимую структурную проверку.

## Структура

```text
data_pipeline/       подготовка holdout и финальных train-корпусов
training/            обучение cross-encoder и финальные recipes
solutions/           runtime, markers, assets и SHA manifests
models/              checkpoint после скачивания Release
submissions/         финальные и пересобранные ZIP
scripts/             загрузка данных, артефактов, сборка и проверка
presentation/        презентация решения
release/             манифест GitHub Release assets
```

Большие файлы не хранятся в Git. Их URL, размеры и SHA-256 находятся в
`release/artifacts.json`, а состав каждого решения — в
`solutions/*/manifest.json`.
