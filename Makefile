PYTHON ?= python3

.PHONY: install test lint fmt

install:
	$(PYTHON) -m pip install -e .
	test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall src tests
