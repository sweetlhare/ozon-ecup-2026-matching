# Подготовка данных

## Входные файлы

Скрипт `scripts/download_data.py` загружает четыре parquet-файла организатора
в каталог `data/`:

| файл | назначение |
|---|---|
| `items.parquet` | полный каталог товарных карточек |
| `items_human.parquet` | товары из ручной части разметки |
| `matches.parquet` | пары с ручной разметкой |
| `matches_llm.parquet` | пары с target, кратным 1/9 |

Размеры, URL и SHA-256 исходных файлов записаны в `raw_data.json`. Загрузчик
проверяет каждый файл и умеет продолжать прерванную передачу.

Для полной цепочки нужны все четыре файла. `matches.parquet` и
`items_human.parquet` участвуют в gold-фазе и в проверке отсутствия пересечений.

## Запуск

Из корня репозитория:

```bash
.venv/bin/python scripts/build_training_data.py --download
```

Если официальные parquet уже лежат в `data/`:

```bash
.venv/bin/python scripts/build_training_data.py
```

Сценарий строит последние train-файлы, ранние closure-корпусы и два больших
retrieval-пула, с которых начинается обучение foundation-моделей.

### Retrieval-пулы foundation

`build_candidate_pool.py` оставляет пары, у которых хотя бы один товар имеет
положительного кандидата в LLM-графе. Так обычные случайные негативы заменяются
на retrieval-негативы того же типа, что использовались при формировании теста.

| префикс | пары | назначение |
|---|---:|---|
| `llmcand` | 6 000 000 | исходный 60k RuModernBERT teacher |
| `llmcand2` | 5 750 952 | 220k scale-модели RuModernBERT и mmBERT |

Для `llmcand2` предварительно исключаются все товары component-level LLM
holdout. `llmcand` повторяет раннюю ветку до введения этого исключения.

`build_novelty_validation.py` отдельно создаёт `val_maxsim.npy`: для каждого
товара ручного validation-сплита хранится максимальная близость к train-товарам
его категории. Этот срез используется только для выбора checkpoint, но также
строится автоматически, поскольку он влияет на результат обучения.

### ORDERED LLM-пул

`build_llmfull.py` сначала воспроизводит промежуточный `llmfull_ord`, который
использовался при поиске повторных входов. После исключения holdout-компонент
в каждой категории остаются сначала все пары с `k>=5`, затем до квоты 525 000
добираются остальные пары с seed 42. Тексты карточек рендерятся в ORDERED-виде
непосредственно из `items.parquet`.

| файл | строки | SHA-256 |
|---|---:|---|
| `llmfull_ord_pairs.parquet` | 10 289 587 | `db88b65eee0ec260bad35ef9ea87487d4330fe51b1d97d0989a40ece53b4b76c` |
| `llmfull_ord_items.parquet` | 11 451 316 | `a2eaefd54a55856451e098e315bd7980deddf0676c5a00eaa90af6a54cc42c18` |

### Closure и consensus

`build_closure_data.py` начинает непосредственно с `matches_llm.parquet` и
`items.parquet`:

1. достраивает отсутствующие рёбра в компонентах, где все наблюдаемые рёбра
   имеют `k=9`;
2. использует `llmfull_ord`, собранный предыдущим шагом после component
   holdout и покатегорийной квоты;
3. находит повторные полностью одинаковые ORDERED-входы, независимо
   размеченные как `k=0`, и подставляет ещё не наблюдавшуюся пару item id с теми
   же текстами;
4. удаляет дубли входов, конфликты меток, пересечения с ручной частью и
   исходными LLM-парами;
5. собирает сбалансированный `closure_consensus` на 17 чистых категориях и
   расширенный `closure_big` для gate;
6. для трёх оставшихся категорий добавляет closure-позитивы и только
   наблюдаемые `k=0` негативы;
7. объединяет их в `closure_all20` для dual-head scorer.

`llmfull_ord_items.parquet` не является внешним или скрытым входом: его создаёт
предыдущий шаг из файлов организатора, после чего closure-builder использует
его как обычный проверяемый промежуточный файл.

| префикс | пары | роль |
|---|---:|---|
| `closure_consensus` | 20 400 | исходный 17-категорийный контроль |
| `closure_big` | 123 733 | 408-шаговая специализация gate |
| `closure_gapcats` | 3 600 | три недостающие категории |
| `closure_all20` | 24 000 | joint dual-head scorer |

### Component-level holdout

`make_llm_val.py` строит граф по `id1/id2` и целиком относит компоненты
связности либо в train, либо в holdout. Это исключает пересечение товаров между
обучением и проверкой.

`build_llmval_sample.py` берёт до 5 000 строк каждой категории с seed
`20260818`. Полученный `llmvalS_pairs.parquet` используется для исключения
holdout-товаров из soft-корпуса.

### Soft-label корпус

`build_soft_train.py`:

1. исключает все пары, связанные с товарами holdout;
2. берёт до 4 000 пар на категорию, seed `31337`;
3. сохраняет исходный target `k/9`;
4. формирует упорядоченный текст обеих карточек.

Результат — `soft_train.parquet`, 80 000 строк.

### Transitive hard negatives

`build_anti_train.py` ищет тройки `positive(center, x)` и
`negative(center, y)`. Если пара `(x, y)` отсутствует в исходной разметке,
она становится hard negative. Для каждой категории выбираются 2 000 таких
негативов и столько же положительных пар.

Результат — `anti_train.parquet`, 80 000 строк, баланс классов 50/50.

### Финальная смесь

`build_mix_train.py` объединяет `soft_train` и `anti_train`, оставляет
`text1/text2/target` и перемешивает строки с `random_state=11`.

Результат — `mix_train.parquet`, 160 000 строк.

## Контроль результата

| файл | строки | SHA-256 |
|---|---:|---|
| `soft_train.parquet` | 80 000 | `d969792a4eb7cc21b867f48d79bdffa96622a7f2d59acf7e9521774f9689bfa2` |
| `anti_train.parquet` | 80 000 | `a96c8a68ddbcef473d022b1c89fa0591dd1e14dc967a052aa52b366d9f91595f` |
| `mix_train.parquet` | 160 000 | `7a7e8c7efa54a2cfcfc16fd30babde5d5d50dc95b8a2cac6862703d39a24cd67` |

Сценарий сам проверяет эти SHA-256 и завершает работу с ошибкой при расхождении.
Для `closure_consensus` дополнительно проверяются исходные SHA-256:

- pairs: `5d6f20fa2c42d588e252bf546de92613b586e0ebce8e7dae9670af9e934f6e77`;
- items: `dc6d66dc411560132b06fb3930dc6edfc11cd180b6fedf7ff9aa6c681405d362`.

Финальный `closure_all20`: pairs
`8fb3d44fbe98d353fbc359d357e311a52eba2218ea638af5bb7880c7cfeca62d`,
items `0fc16a96ca9ea485e590a22f831e506e3f2a1d8f5f94369aadf1ac34d190c35f`.

`build_text_negative_signatures.py` группирует полностью одинаковые ORDERED
входы в `llmfull_ord`. Группа попадает в runtime-коррекцию, только если она
встречается не меньше двух раз и все независимые метки равны `k=0`. Результат
`textsig_negative_u64.npy` также проверяется по SHA-256 и используется только
основным `mix408_textneg`.

Для byte-identical parquet нужны версии из `requirements-data.txt`.
