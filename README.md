# sdmodeltester

A small harness for testing **Stable Diffusion 1.5 and SDXL** single-file `.safetensors`
checkpoints from a minimal browser UI. The real runtime is a Google Colab notebook on an
NVIDIA L4: the checkpoint is loaded once, kept resident on the GPU, and driven repeatedly
(text-to-image and image-to-image) through Colab's built-in kernel proxy.

> **Status:** early development. See [`objectives/status.md`](objectives/status.md) for
> current progress and [`objectives/SPEC.md`](objectives/SPEC.md) for the full V1 spec.

## Layout

```text
src/                     application code (backend, prompting, server, CLI, UI, benchmark)
notebooks/               Colab notebook + its helper module
tests/                   test suite — runs without torch/diffusers/GPU
models/ inputs/ outputs/ runtime data (git-ignored)
objectives/              SPEC.md (requirements + decisions), status.md (progress)
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

### Benchmark (Phase 8, run on the L4)

```bash
python -m src.cli benchmark --model-family sdxl --model models/x.safetensors \
    --profiles baseline,compile,compile-max
```

Profiles run one after another, each with a fresh backend: cold load, compile/warm-up,
warm single-image latency (×3), batch 5/10, a new resolution, img2img, peak VRAM. The first
profile also runs the functional matrix (all samplers, seeds, prompt syntax, img2img,
batching, black-image detection). Output: `compat/benchmark-<family>-<date>.md` with a
recommended default profile, plus a contact sheet in `outputs/benchmark/` for visual review.
On Colab, run it from a terminal or a scratch cell in `/content/sdmodeltester` after stopping
the UI server (it needs the VRAM).

Unset knobs (`--width`, `--height`, `--steps`, `--cfg`, `--sampler`, `--strength`) use the
model family's defaults. `generate` prints output paths on stdout and progress on stderr.
Exit codes: 0 success, 1 runtime/load failure (or `doctor` found a missing required
capability), 2 invalid arguments or request.

## Run on Colab (L4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwin2rai/sdmodeltester/blob/main/notebooks/colab.ipynb)

`notebooks/colab.ipynb` is a minimal, shareable orchestrator (4 small cells): settings →
clone this repo + install app libraries on top of Colab's own torch + `doctor` → download
one checkpoint → `python -m src.cli serve` in the background with the UI embedded through
Colab's built-in kernel proxy. It prints one ✓ line per step and nothing from the UI; the
server log is in `/content/sdmodeltester/server.log`.

- `MODEL_URL`: a Hugging Face file link (`hf_hub_download`) or a Civitai link that includes
  the version (`?modelVersionId=…` or `/api/download/models/<id>`); the file must be a
  full-size `.safetensors` checkpoint.
- Tokens: Colab Secrets `HF_TOKEN` / `CIVITAI_TOKEN` are used first, so shared copies never
  contain anyone's token.

Helpers live in `notebooks/colab_utils.py` (notebook-only, never imported by `src/`).
`requirements-inference.txt` intentionally does **not** list torch — see
[`objectives/SPEC.md`](objectives/SPEC.md) §3.

## License

MIT
