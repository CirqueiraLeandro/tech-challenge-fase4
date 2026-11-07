.PHONY: install dev download-data validate-env lint format test prepare train bad-batch validate-bad \
        simulate monitor causal fairness report retrain all demo obs-up obs-down retention

PY = poetry run python
CLI = $(PY) -m src.pipeline.cli

install:
	poetry install --only main

dev:
	poetry install --with dev

download-data:
	$(PY) -m scripts.download_data

validate-env:
	$(PY) -m scripts.validate_env

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

test:
	$(PY) -m pytest -m "not requires_data" --cov=src --cov-report=term

prepare:
	$(CLI) prepare

train:
	$(CLI) train

bad-batch:
	$(PY) -m scripts.make_bad_batch

validate-bad:
	$(CLI) validate data/incoming/bad_batch.csv

simulate:
	$(CLI) simulate

monitor:
	$(CLI) monitor

causal:
	$(CLI) causal

fairness:
	$(CLI) fairness

report:
	$(CLI) report

retrain:
	$(CLI) retrain --with-windows M4 M5

all:
	$(CLI) all

demo:
	$(CLI) demo --synthetic

obs-up:
	docker compose up -d

obs-down:
	docker compose down

retention:
	$(PY) -m scripts.apply_retention
