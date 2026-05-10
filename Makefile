.PHONY: install test lint format check reproduce

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

check:
	ruff check .
	ruff format --check .
	pytest

reproduce:
	scripts/reproduce.sh
