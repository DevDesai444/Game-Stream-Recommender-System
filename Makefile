.PHONY: help install dev test test-cov lint format type benchmark loadtest demo data clean

PY ?= python3

help:
	@echo "Targets:"
	@echo "  install     install runtime deps"
	@echo "  dev         install runtime + dev deps + pre-commit hooks"
	@echo "  data        download the Steam-200k dataset (~8.5 MB)"
	@echo "  data-ucsd   download the UCSD Steam dataset (~80 MB)"
	@echo "  test        run unit tests"
	@echo "  test-cov    run unit tests with coverage (gate at 73%)"
	@echo "  lint        ruff + black --check"
	@echo "  format      ruff --fix + black"
	@echo "  type        mypy on the package"
	@echo "  benchmark   run the full hybrid benchmark on Steam-200k"
	@echo "  benchmark-ucsd  run the full hybrid (with two-tower NCF) on UCSD"
	@echo "  loadtest    measure FastAPI P95 against an in-process worker"
	@echo "  demo        boot the Streamlit demo"
	@echo "  clean       drop caches and generated artifacts"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

dev: install
	$(PY) -m pip install -r requirements-dev.txt
	pre-commit install --install-hooks

data:
	bash scripts/download_dataset.sh

data-ucsd:
	bash scripts/download_ucsd_dataset.sh

test:
	PYTHONPATH=src $(PY) -m pytest tests/unit -q

test-cov:
	PYTHONPATH=src $(PY) -m pytest tests/unit --cov=src/gamereco --cov-report=term --cov-fail-under=73

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

type:
	mypy src/gamereco

benchmark:
	PYTHONPATH=src $(PY) scripts/run_benchmark.py --out benchmarks/results.json

benchmark-ucsd:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src $(PY) scripts/run_benchmark_ucsd.py \
		--max-users 4000 --als-factors 32 --als-iters 8 \
		--kmeans-k 12 --candidates 150 --two-tower-epochs 3 \
		--two-tower-embedding-dim 24 --two-tower-output-dim 24 \
		--out benchmarks/results_ucsd.json

loadtest:
	PYTHONPATH=src $(PY) scripts/run_loadtest.py --requests 2000 --concurrency 16

demo:
	streamlit run demo/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
