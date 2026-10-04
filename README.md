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

Two optional test layers (each skipped automatically when not installed):

```bash
# Real Diffusers backend on CPU with tiny random SD1.5/SDXL pipelines built from configs
# (no checkpoint, no network; CPU-only torch ~190 MB).
uv run --group inference-cpu pytest

# Browser tests of the UI in headless Chromium (one-time browser install).
uv run --group ui playwright install chromium --only-shell
sudo "$(pwd)/.venv/bin/playwright" install-deps chromium-headless-shell   # system libs, once
uv run --group ui pytest tests/test_ui.py

uv run --all-groups pytest      # everything (~45 s)
```

Note: a plain `uv sync` / `uv run` removes optional groups again (uv syncs exactly).
`requirements.txt` / `requirements-inference.txt` remain for the Colab notebook, which
installs on top of Colab's own Python and PyTorch.

## CLI

```bash
uv run python -m src.cli doctor --mock                       # mock-mode requirements only
uv run python -m src.cli doctor --model models/x.safetensors --model-family sdxl --compile-check
uv run python -m src.cli generate --mock --model-family sd15 --prompt "a {white | black} cat" \
    --seed 123 --images 4 --output-dir outputs
uv run python -m src.cli generate --model-family sdxl --model models/x.safetensors \
    --prompt "portrait, (white hair:1.2)" --image inputs/src.png --strength 0.6
uv run python -m src.cli serve --mock --model-family sdxl --port 8000    # open http://127.0.0.1:8000
```

### Benchmark (Phase 8, run on the Colab L4)

```bash
python -m src.cli benchmark --model-family sdxl --model models/x.safetensors \
    --profiles baseline,compile,compile-max
```

Each profile runs in a fresh process: cold load, compile/warm-up, warm single-image
latency (×3), batch 5/10, a new resolution, img2img, and peak VRAM. The verify profile
(default: the first) also runs the functional matrix (all samplers, seeds, prompt syntax,
img2img, batching, black-image detection). Output: `compat/benchmark-<family>-<date>.md`
(+ `.json`) with a recommended default profile, and a contact sheet of every check image
in `outputs/benchmark/` for visual review. `--mock` exercises the same flow without a GPU.

Unset knobs (`--width`, `--height`, `--steps`, `--cfg`, `--sampler`, `--strength`) use the
model family's defaults. `generate` prints output paths on stdout and progress on stderr.
Exit codes: 0 success, 1 runtime/load failure (or `doctor` found a missing required
capability), 2 invalid arguments or request.

## Real inference (Colab L4)

`requirements-inference.txt` holds the Diffusers-side dependencies. It intentionally does
**not** list torch: Colab supplies Python, CUDA, and PyTorch. The Colab notebook is built
last, after the backend, server and UI are stable — see
[`objectives/COLAB_COMPATIBILITY.md`](objectives/COLAB_COMPATIBILITY.md).

## License

MIT
