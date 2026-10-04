# sdmodeltester

A small harness for testing **Stable Diffusion 1.5 and SDXL** single-file `.safetensors`
checkpoints from a minimal browser UI. The real runtime is a Google Colab notebook on an
NVIDIA L4: the checkpoint is loaded once, kept resident on the GPU, and driven repeatedly
(text-to-image and image-to-image) through Colab's built-in kernel proxy.

> **Status:** early development. See [`objectives/status.md`](objectives/status.md) for
> current progress and [`objectives/SPEC.md`](objectives/SPEC.md) for the full V1 spec.

## Layout

```text
src/                     application code (backend, prompting, server, CLI, UI)
tests/                   test suite — runs without torch/diffusers/GPU
models/ inputs/ outputs/ runtime data (git-ignored)
objectives/              spec, Colab compatibility policy, development status
```

## Local development (mock mode, no GPU)

Local development uses [uv](https://docs.astral.sh/uv/) and Python 3.13 (Colab's current
default runtime; pinned in `.python-version`).

```bash
uv sync                  # creates .venv with app + dev dependencies
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The normal test suite never needs torch, diffusers, CUDA, or a model download.

Optionally, the real Diffusers backend can be exercised on CPU with tiny random SD1.5/SDXL
pipelines built from configs (no checkpoint or network; CPU-only torch, ~190 MB):

```bash
uv run --group inference-cpu pytest     # adds tests/test_diffusers_cpu.py (~15 s)
```

Note: a plain `uv sync` / `uv run` removes the optional group again (uv syncs exactly).
`requirements.txt` / `requirements-inference.txt` remain for the Colab notebook, which
installs on top of Colab's own Python and PyTorch.

Planned commands (not implemented yet):

```bash
uv run python -m src.cli doctor
uv run python -m src.cli serve --mock --model-family sdxl --port 8000
uv run python -m src.cli generate --mock --model-family sd15 --prompt "a {white | black} cat"
```

## Real inference (Colab L4)

`requirements-inference.txt` holds the Diffusers-side dependencies. It intentionally does
**not** list torch: Colab supplies Python, CUDA, and PyTorch. The Colab notebook is built
last, after the backend, server and UI are stable — see
[`objectives/COLAB_COMPATIBILITY.md`](objectives/COLAB_COMPATIBILITY.md).

## License

MIT
