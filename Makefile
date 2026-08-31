PYTHON ?= .venv/bin/python

.PHONY: setup data artifacts train train-final submissions verify all

setup:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/build_training_data.py --download

artifacts:
	$(PYTHON) scripts/download_artifacts.py

train:
	PYTHON=$(PYTHON) bash training/train_from_scratch.sh

train-final:
	PYTHON=$(PYTHON) bash training/train_full_pipeline.sh

submissions:
	$(PYTHON) scripts/submission.py build mix408_textneg --force
	$(PYTHON) scripts/submission.py build final_safe_fwdgate --force

verify:
	$(PYTHON) scripts/submission.py sources mix408_textneg
	$(PYTHON) scripts/submission.py sources final_safe_fwdgate
	$(PYTHON) scripts/submission.py verify mix408_textneg submissions/submission_mix408_textneg.zip
	$(PYTHON) validate_submission.py submissions/rebuilt_submission_final_safe_fwdgate.zip --expect-models 2 --expect-ordered model1_nosym model2_nosym --require-fp16

all: data artifacts submissions verify
