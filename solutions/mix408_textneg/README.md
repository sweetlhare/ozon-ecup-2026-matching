# mix408_textneg

Основной финалист и лучший отправленный public-сабмит: **0.5480056269**.

- отправленный архив: `submissions/submission_mix408_textneg.zip`;
- SHA-256 архива:
  `40eaad4f58a6c7266944d57e1c7b2d606f70abd2f0e62a2cb1ff66521c832e1c`;
- gate: `sgate_670`, симметричный, `model1/`;
- scorer: `mix408`, `model2_nosym/`, partial symmetry `0.75`;
- fusion: rank-based disagreement, `RRF_K=100`;
- exact negative text-consensus: `textsig_negative_u64.npy`;
- entry point: `python -u run.py --cascade 0.65 --max_len 640 --workers 2`.

`runtime/`, `markers/` и `assets/` извлечены прямо из отправленного ZIP.
Checkpoint публикуются в GitHub Release и после скачивания лежат в
`models/mix408_textneg/{gate,scorer}`. Полный состав и SHA каждого member — в
[`manifest.json`](manifest.json).

```bash
python3 scripts/submission.py verify mix408_textneg \
  submissions/submission_mix408_textneg.zip
python3 scripts/submission.py build mix408_textneg
```

Первая команда требует byte-identical отправленный архив. Вторая создаёт
детерминированную content-identical пересборку под другим именем; внешний SHA
ZIP из-за новых ZIP metadata не обязан совпасть с отправленным.
