# Submission archives

ZIP-файлы не входят в обычный Git из-за размера, но опубликованы в GitHub
Release `reproducibility-v1` и скачиваются командой:

```bash
python3 scripts/download_artifacts.py --only submissions
```

- `submission_mix408_textneg.zip` — точный отправленный основной архив;
- `rebuilt_submission_final_safe_fwdgate.zip` — проверенная content-identical
  пересборка runtime-хеджа, не исходный байт-в-байт ZIP; SHA-256 пересборки
  `e33777770bd1aa9f4619939e866f429814ed0d9c0cf89eb7b1d1f386184cc6a0`.

Внешние SHA отправленных архивов и SHA каждого member хранятся в
`release/artifacts.json` и `solutions/*/manifest.json`.
