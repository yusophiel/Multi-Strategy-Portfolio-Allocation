.PHONY: install run test

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m src run --config configs/default.json --output outputs/default_run

test:
	$(PYTHON) -m unittest discover -s tests
