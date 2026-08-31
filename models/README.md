# Local model store

Этот каталог не версионируется. Сборщик ожидает четыре каталога:

```text
models/mix408_textneg/gate/
models/mix408_textneg/scorer/
models/final_safe_fwdgate/gate/
models/final_safe_fwdgate/scorer/
```

Скачивание всех четырёх checkpoint:

```bash
python3 scripts/download_artifacts.py --only models
```

Точные SHA-256 всех обязательных файлов находятся в
`solutions/*/manifest.json`; команда `python3 scripts/submission.py sources …`
проверяет их после скачивания и до сборки.
