# Common developer tasks. Requires `make` (available via git-bash/WSL on Windows).

.PHONY: help install test lint format typecheck docs bench tune media gui dashboard clean

help:
	@echo "Targets: install test lint format typecheck docs bench tune media gui dashboard clean"

install:        ## install the package with all extras
	python -m pip install -e ".[all]" && python -m pip install cmaes

test:           ## run the test suite
	pytest

lint:           ## lint + import order
	ruff check src tests benchmarks scripts examples apps

format:         ## auto-format
	ruff format src tests benchmarks scripts examples apps

typecheck:      ## static type check (non-blocking)
	mypy src || true

docs:           ## serve the documentation site locally
	mkdocs serve

bench:          ## run the full controller benchmark
	python benchmarks/run_all.py

tune:           ## auto-tune every controller and report the improvement
	python benchmarks/tune_all.py

media:          ## regenerate the README/demo media
	python examples/generate_media.py && python examples/generate_extra_media.py

gui:            ## launch the desktop control center
	python apps/desktop_gui.py

dashboard:      ## launch the web dashboard
	streamlit run apps/streamlit_app.py

clean:          ## remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info site
	find . -type d -name __pycache__ -not -path "./Control GUI/*" -exec rm -rf {} +
