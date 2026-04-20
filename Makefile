.PHONY: install sync lint format test data eda train tune evaluate all clean

PY := uv run

install:
	uv sync --extra dev

sync:
	uv sync --extra dev

lint:
	$(PY) ruff check src tests

format:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

test:
	$(PY) pytest

data:
	$(PY) teen-mh ingest

eda:
	$(PY) teen-mh eda

train:
	$(PY) teen-mh train

tune:
	$(PY) teen-mh tune

evaluate:
	$(PY) teen-mh evaluate

all: data eda train tune evaluate

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist mlruns
	find . -type d -name __pycache__ -exec rm -rf {} +
