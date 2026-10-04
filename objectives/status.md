# Development Status & Learnings

This file is the persistent memory for development sessions. Read it first when resuming.
Update it at every pause point. Source-of-truth specs: `SPEC.md`, `COLAB_COMPATIBILITY.md`.

**Last updated:** 2026-10-04

---

## Working agreement (from the repo owner)

- **Memory lives here**, in `objectives/status.md` — not in temp folders or agent-private memory.
- **Never commit or push.** Stop at natural pause points; the owner reviews, tests, and commits.
- **Work in small steps** with pauses between them — don't one-shot the project.
- **Dev machine is a small Codespace** (2 CPU, ~8 GB RAM, no GPU). Don't download large
  *data*: no SD checkpoints, no big test images (keep any test data to a few MB at most;
  prefer generating images in-test with Pillow).
- **Libraries/tools are fine:** install Python packages and dev tools freely without asking;
  make the best decision. (Still never install torch/diffusers here — no GPU, and they are huge.)
- **Use uv** for Python environment/dependency management.
- **Priority:** build and test everything that works in mock / representative tests first.
  Real-GPU work (Phase 8) and the Colab notebook (Phase 9) come last and need Colab.

## Environment notes

- **Colab now defaults to Python 3.13** (upgraded from 3.12.13 to 3.13.15; owner note
  2026-10-04). Local dev is pinned to 3.13 via `.python-version` (uv-managed CPython 3.13.16).
  `requires-python = ">=3.10"`; ruff `target-version = "py310"` keeps syntax portable.
  Note: SPEC/COLAB_COMPATIBILITY mention 3.12 — that is outdated on this point.
- Colab release 2026-09-16 (owner FYI): Python 3.12.13 → 3.13.15 (all images); Ubuntu
  22.04 → 24.04 (24.04.1 on GPU images); tokenizers 0.22.2 → 0.23.1 (relevant to the
  transformers bounds in `requirements-inference.txt`); also jax 0.11.1, numba 0.61.2,
  pandas 2.2.3. The torch version wasn't listed; `doctor` / Phase 8 must record it.
- Setup: `uv sync` then `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.
- Dev deps are in `[dependency-groups] dev` (pytest, ruff). App deps (flask, pillow) are in
  `[project] dependencies` and mirrored in `requirements.txt` for the Colab notebook.
  `uv.lock` is generated and should be committed.
- No `[build-system]`: uv treats the project as non-packaged; pytest finds `src` via
  `pythonpath = ["."]`. Run modules as `uv run python -m src.cli ...`.
- Ruff now formats Markdown code blocks too; `extend-exclude = ["objectives", "*.md"]`
  stops it from reformatting the spec docs.

## Phase progress

| Step | Description | State |
|---|---|---|
| 0 | Repo setup (layout, manifests, pyproject, README, gitignore, status) | done (owner committing) |
| 1 | Core pure-Python: validation, family literal, seeds, dynamic prompts, preprocess, filenames | done |
| 1b | Owner corrections: round dims to ×8, quietly raise img2img steps | done |
| 2 | Mock backend + shared MicroBatcher (OOM fallback) | **done — awaiting owner review/commit** |
| 3 | Real Diffusers backend (code only; lazy imports; untestable here beyond import guards) | not started |
| 4 | CLI: doctor / generate / serve | not started |
| 5 | Queue + HTTP API | not started |
| 6 | UI (`src/static/index.html`) | not started |
| 7 | Local/mock verification | not started |
| 8 | L4 tuning + `compat/known-good-colab.md` | needs Colab |
| 9 | Colab notebook | needs Colab, last |

## What exists (Phase 1)

- `src/prompting.py`: `parse_template`, `validate_template`, `resolve_template`,
  `resolve_prompt_pair(prompt, negative, seed)`, `prompt_rng(seed)` (seed XOR
  `PROMPT_RNG_SALT`), `PromptSyntaxError`. Weighted-embedding adapter not yet written.
- `src/backend.py`: `ModelFamily`, `MODEL_FAMILIES`, `check_family`, `SAMPLERS` /
  `SAMPLER_IDS` / `SAMPLER_LABELS`, `FAMILY_DEFAULTS`, `GenerationRequest` (+ `.mode`),
  `ResolvedImageSpec`, `ResolvedGenerationJob` (+ `job_id`, `.seeds`), `GenerationResult`,
  `ValidationError`, `round_dimension`, `min_img2img_steps`, `normalize_request`,
  `validate_request`, `resolve_seeds`, `resolve_job`,
  `preprocess_image`, `output_filename`, `collision_safe_path`.
- Phase 2 additions in `src/backend.py`: `Backend` Protocol (`family`, `model_name`, `load`,
  `warmup`, `generate`), `StatusCallback = (state, message)`,
  `ProgressCallback = (fraction 0..1, message)`, `GenerationError(message, completed_paths)`,
  `denoising_steps(req)`, `MicroBatcher`, `MockBackend`, `MockOutOfMemory`, `save_png`.
- Tests: `test_package.py`, `test_prompting.py`, `test_seeds.py`, `test_request.py`,
  `test_preprocess.py`, `test_mock_backend.py` — 98 passing. `make_request()` helper lives in `test_seeds.py`.

## Decisions made

- pytest `addopts` deselects `-m gpu`; real-backend tests must carry `@pytest.mark.gpu`.
- `test_package.py` asserts `import src` doesn't import torch/diffusers/transformers
  (subprocess, so other tests can't pollute `sys.modules`).
- `compat/` intentionally not created yet — only after real L4 verification.
- Runtime dirs `models/ inputs/ outputs/` git-ignored except `.gitkeep`; `*.safetensors`,
  `*.ckpt` ignored repo-wide.
- **Request pipeline**: `resolve_job` = `normalize_request` (silent corrections, returned
  as `job.corrections` notes for the log) → `validate_request` (strict checks) → seeds →
  dynamic prompts.
- **Dimensions** (owner decision): rounded to the nearest multiple of 8, ties up
  (516→520, 1001→1000, 252→256), *then* the hard range 256–2048 is checked; out of range
  is rejected, not clamped. UI uses step 64.
- **Other limits**: steps 1–150; CFG 0–30; images 1–10; prompt ≤ 4000 chars; seed `-1` or
  0..2³²−1, and `seed + n − 1` must not exceed 2³²−1 (rejected, no wraparound; owner OK'd).
- **img2img steps** (owner decision, A1111-like): Diffusers runs `int(steps × strength)`
  denoising steps and errors on 0. If that's 0, quietly raise steps to the minimum giving
  1 denoising step (5 @ 0.1 → 10) and add a note. If that would need >150 steps
  (strength < ~0.0067), reject. Strength must be in (0, 1]; strength 0 is rejected.
  No A1111 "exact steps" mode in V1.
- **Strength for txt2img** must be `None` (owner OK'd); the server/CLI drop it.
- **Random seeds** (`-1`): drawn with `SystemRandom`, guaranteed unique within a job.
- **Dynamic prompts**: one PRNG per image seeded from that image's seed; positive resolved
  first, then negative, from the same PRNG. An image's choices depend only on its own seed
  (seed 124 resolves identically whether it's image 1 or image 2 of a job). Owner confirmed:
  images in a fixed-seed job get base+0, base+1, … and each resolves its own dynamic
  choice, so images may (randomly) differ in their picks.
- Stray `}` outside a group and unclosed `{` are syntax errors. `|` outside braces is literal.
- `job_id` and `corrections` live on `ResolvedGenerationJob` (defaults 0 and `()`).
- Filenames: `YYYYMMDD_HHMMSS_job0007_img01_seed123.png` (`img` index is 1-based);
  `collision_safe_path` appends `_1`, `_2`… if needed.
- Preprocess: cover scale = max(w/sw, h/sh), LANCZOS, center crop; accepts a path or a PIL image.

- **MicroBatcher** (shared by mock + real): `run(key, items, run_batch, is_oom, progress_callback,
  on_oom, on_batch_done)`; key = `(mode, width, height)`. Tries the whole job (capped by a
  learned limit); on OOM halves; a limit is stored only after an OOM, and only from a
  *full-size* success (a smaller remainder batch never lowers it). Limits persist per
  backend instance (process lifetime). OOM at batch size 1 or a non-OOM error re-raises.
  `run_batch(batch, on_step)` must call `on_step(step, total_steps)` each denoising step.
- **Progress message**: `"Batch 1/2 · denoising 14/25"` when >1 batch, else
  `"denoising 14/25"`; fraction advances continuously across batches. The batch count is
  recomputed after an OOM. The server adds the "Job N:" prefix.
- **img2img progress total** = `int(steps × strength)` (`denoising_steps`), the number of
  steps actually run.
- **Mock backend knobs**: `load_seconds`, `step_seconds`, `max_batch` (simulated OOM),
  `fail_load`, `fail_after_images`; `batch_sizes` records attempted sizes. States emitted:
  loading → optimizing (in `load`), warming (in `warmup`). `generate` before `load` raises.
  Output = flat colour from the seed (txt2img), or the preprocessed input blended with that
  colour (img2img), with family/mode/index/seed/prompt drawn as text (pixels only).
- **save_png** copies pixels into a fresh image, so no text/EXIF/ICC metadata carries over.
  The test checks for no tEXt/iTXt/zTXt/eXIf chunks.
- One filename timestamp per job (all images in a job share it).
- pillow floor raised to `>=10.1` (`ImageFont.load_default(size=)`).

## Open questions / TODO

- Weighted-prompt library choice (spec mentions `sd_embed`) — decide in Phase 3; keep behind
  `src/prompting.py`. A pure-Python `(text:weight)` parser could avoid the extra dependency.
- SPEC/COLAB_COMPATIBILITY still reference Python 3.12 — consider updating docs.

## Learnings

- Ruff ≥ 0.16 formats code blocks inside Markdown — exclude docs from ruff.
- Pillow EXIF test: save a JPEG with `exif[0x0112] = 6`; `ImageOps.exif_transpose` rotates it.
- Bug caught by tests: the remainder batch (e.g. the final 1 of 7 at size 3) originally
  overwrote the learned OOM limit with 1. Fixed: only full-size successes update the limit.
- A deliberate "stretch instead of crop" bug makes `test_preprocess.py` fail, so the
  no-distortion tests do catch that regression.
