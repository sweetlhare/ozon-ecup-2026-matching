# final_safe_fwdgate

Фактически отправленный runtime-хедж: **0.5426350252** public.

- исходный архив: `submission_final_safe_fwdgate.zip`;
- SHA-256 исходного архива:
  `108249a36ad5de60b27f99027b18c1c79828f4b1981560f8dd08b9c46d769d5d`;
- gate: исходный `a10_gate_parent`, forward-only, `model1_nosym/`;
- scorer: `a10_scorer_parent`, `model2_nosym/`, partial symmetry `0.75`;
- fusion: rank-based disagreement, `RRF_K=100`;
- entry point: `python -u run.py --cascade 0.65 --max_len 640 --workers 2`.

Этот архив был выбран как страховка от private timeout: он убирает обратный
проход gate ценой измеренной public-дельты. Это не попытка улучшить основной
score.

В Release опубликована детерминированная пересборка. Все 19 файлов внутри неё
совпадают с финальным manifest по SHA-256. Внешний SHA исходного ZIP отличается
из-за ZIP metadata; SHA-256 реально отправленного архива приведён выше.

```bash
python3 scripts/submission.py sources final_safe_fwdgate
python3 scripts/submission.py build final_safe_fwdgate
```

