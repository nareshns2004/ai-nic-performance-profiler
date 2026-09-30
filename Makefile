PYTHON ?= python3
export PYTHONPATH := src

.PHONY: install dev test lint typecheck fmt check demo eval bench docker clean

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests benchmarks examples
	ruff format --check src tests benchmarks examples

typecheck:
	mypy

fmt:
	ruff check --fix src tests benchmarks examples
	ruff format src tests benchmarks examples

check: lint typecheck test

demo:
	$(PYTHON) -m ai_nic_perf_profiler demo

eval:
	$(PYTHON) -m ai_nic_perf_profiler evaluate --seeds 5

bench:
	$(PYTHON) benchmarks/run_benchmarks.py

docker:
	docker build -t nicprof:latest .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache run
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
