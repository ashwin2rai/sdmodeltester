# Short commands for local development. Run `make` (or `make help`) to list them.
# Override variables on the command line, e.g. `make serve-mock FAMILY=sdxl PORT=8001`.

FAMILY ?= sd15
PORT   ?= 8000
MODEL  ?=

# --inexact: don't strip optional groups (e.g. CPU torch from `make test-all`) on every run
RUN := uv run --inexact --group ui

.DEFAULT_GOAL := help
.PHONY: help setup setup-all test test-all lint format check serve-mock serve doctor

help:  ## list commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup:  ## install app + dev + UI-test deps and headless Chromium
	uv sync --group ui
	$(RUN) playwright install chromium --only-shell
	@echo "First time on this machine? sudo .venv/bin/playwright install-deps chromium-headless-shell"

setup-all: setup  ## also install the CPU torch + diffusers stack (tiny-model tests)
	uv sync --all-groups

test:  ## fast test suite (no torch needed; UI tests run if Chromium is installed)
	$(RUN) pytest -q

test-all:  ## every local test group, incl. tiny CPU pipelines (~75 s)
	uv run --all-groups pytest -q

lint:  ## ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## auto-fix lint issues and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

check: lint test  ## lint + tests (run before committing)

serve-mock:  ## UI with the mock backend at http://127.0.0.1:8000 (no GPU)
	$(RUN) python -m src.cli serve --mock --model-family $(FAMILY) --port $(PORT)

serve:  ## UI with a real checkpoint: make serve MODEL=models/x.safetensors FAMILY=sdxl
	@test -n "$(MODEL)" || { echo "MODEL=path/to/checkpoint.safetensors is required"; exit 2; }
	$(RUN) python -m src.cli serve --model-family $(FAMILY) --model $(MODEL) --port $(PORT)

doctor:  ## check Python/torch/CUDA/GPU (fails without torch, e.g. here)
	$(RUN) python -m src.cli doctor
