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
  make the best decision. CPU-only torch + diffusers are installed via the optional
  `inference-cpu` group for tiny-model tests (never the CUDA torch build: several GB).
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
  Everything incl. optional groups: `uv run --all-groups pytest` (~70 s, 247 tests at Phase 8a).
- **Optional UI test stack**: `ui` group = playwright (1.63). Chromium headless shell in
  `~/.cache/ms-playwright` (~270 MB) via `playwright install chromium --only-shell`; the
  Codespace needed system libs once: `sudo .venv/bin/playwright install-deps
  chromium-headless-shell` (sudo works without a password here).
- **Optional CPU inference stack**: `uv run --group inference-cpu pytest` (torch 2.14.1+cpu,
  diffusers 0.40.0, transformers 5.18.0 at time of writing). torch comes from the
  `pytorch-cpu` index (`[tool.uv.sources]`), *never* PyPI's CUDA build. uv syncs exactly,
  so a plain `uv run`/`uv sync` uninstalls the group; reinstalling takes ~40 s from cache.
  To check the torch-free suite without disturbing `.venv`:
  `UV_PROJECT_ENVIRONMENT=<scratch>/venv-notorch uv run pytest`.
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
| 0 | Repo setup (layout, manifests, pyproject, README, gitignore, status) | committed |
| 1 | Core pure-Python: validation, family literal, seeds, dynamic prompts, preprocess, filenames | committed |
| 1b | Owner corrections: round dims to ×8, quietly raise img2img steps | committed |
| 2 | Mock backend + shared MicroBatcher (OOM fallback) | committed |
| 3 | Real Diffusers backend + weighted prompts; CPU tiny-model tests | committed; Colab-only parts pending Phase 8 |
| 4 | CLI: doctor / generate / serve | committed |
| 5 | Queue + HTTP API | committed |
| 6 | UI (`src/static/index.html`) + headless-browser tests | committed (owner checked the UI by hand: OK) |
| 7 | Local/mock verification | effectively done via automated tests (mock, CPU-tiny, browser) |
| 8a | Phase 8 tooling: `benchmark` CLI (profiles + functional matrix + report) | **done — awaiting owner review/commit** |
| 8b | Run the benchmark on a real L4 for SD1.5 + SDXL, pick the default profile, write `compat/known-good-colab.md` | needs Colab |
| 9 | Colab notebook | needs Colab, last |

## What exists

- `src/prompting.py`: `parse_template`, `validate_template`, `resolve_template`,
  `resolve_prompt_pair(prompt, negative, seed)`, `prompt_rng(seed)` (seed XOR
  `PROMPT_RNG_SALT`), `PromptSyntaxError`. Phase 3: `parse_weighted`, `weighted_token_ids`,
  `chunk_count`, `chunk_tokens`, `CHUNK_TOKENS` (pure) and `encode_prompt_batch(pipe, family,
  prompts, negatives)` (torch, lazy) returning the pipeline embedding kwargs.
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
- Phase 3 additions in `src/backend.py`: `SAMPLER_SCHEDULERS`, `OPTIMIZATION_PROFILES`,
  `DEFAULT_OPTIMIZATION`, `WARMUP_STEPS`, `configure_cuda_allocator()`, `make_scheduler()`,
  `DiffusersBackend(family, model_path, device, dtype, optimization, warmup_steps,
  warmup_size, log, pipeline_loader)`.
- `src/cli.py` (Phase 4): `main(argv, out, err) -> exit code`, `build_parser`,
  `build_backend(args, log, mock_delays)`, `build_request(args)`, `check_model_path`,
  `collect_diagnostics(mock, model, family, compile_check) -> list[Check]`, `format_checks`,
  `cmd_doctor` / `cmd_generate` / `cmd_serve`, `UsageError`. `serve` calls
  `src.server.serve(backend, host, port, inputs_dir, outputs_dir)`; Phase 5 must provide it.
- `src/benchmark.py` (Phase 8a): `benchmark_profile(backend, family, out, profile, steps, size,
  alt_size, log)`, `functional_checks(...) -> list[Check]`, `contact_sheet`, `recommend`,
  `render_report`, `is_black`, `max_pixel_diff`, `default_report_name`. CLI: `cmd_benchmark`
  (+ `_run_profile`, `_run_profile_subprocess`, `_strip_profile_args`; hidden `--child-json`).
  Backends gained `load_count`; DiffusersBackend gained `reset_peak_memory()` / `memory_stats()`.
- `src/static/index.html` (Phase 6): the whole UI, inline CSS/JS, no build step.
- `src/server.py` (Phase 5): `ServerState(backend, inputs_dir, outputs_dir, max_queue, echo)`
  with `start/stop/enqueue/clear_queue/status/log/log_since/is_busy`; `JobRecord`;
  `create_app(state)`; `serve(backend, host, port, inputs_dir, outputs_dir)`;
  `request_from_json`, `list_files`, `safe_child`, `secure_delete`, `clear_directory`,
  `format_seeds`; `MAX_QUEUE = 5`. `src/static/index.html` is a placeholder.
- Tests: `test_benchmark.py`, `test_ui.py` (ui marker; Playwright), `test_server.py`, `test_cli.py`, `test_package.py`, `test_prompting.py`, `test_seeds.py`, `test_request.py`,
  `test_preprocess.py`, `test_mock_backend.py`, `test_real_backend.py` (no torch), and
  `test_diffusers_cpu.py` (torch marker; skipped without the group). 108 pass without torch
  (1 module skipped); 210 pass with `--group inference-cpu` in ~18 s.
  `make_request()` helper lives in `test_seeds.py`.

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

### Phase 3 decisions

- **Prompt weighting is implemented in-house** (not `sd_embed`, which isn't on PyPI and
  would mean a git dependency on Colab). Method: A1111 "original" emphasis, applied per
  77-token chunk: z *= weight per token, then rescale so the chunk mean is unchanged.
  Unweighted prompts skip this, so they exactly match diffusers' `encode_prompt`; tests
  assert that for SD1.5 and SDXL. SD1.5 uses `last_hidden_state` (clip_skip None); SDXL uses
  `hidden_states[-2]` of both encoders concatenated, and pooled = `text_encoder_2` output[0]
  of the first chunk.
- **Weight syntax**: regex `\(([^()]*):\s*number\s*\)`; other parentheses are literal;
  `\(`/`\)` are escapes. `(ratio:16:9)` parses as "ratio:16" weight 9 (same as A1111). No
  nesting, no bare `(x)` = 1.1, no `[x]`.
- **Long prompts**: 75-token chunks `[BOS] + ≤75 + [EOS] + PAD`; every prompt in a batch
  (positive and negative) is padded to the same chunk count with empty chunks. Each distinct
  string is encoded once per batch (10 identical prompts → 1 positive + 1 negative encode).
- **SDXL empty negative** → zero embeddings and zero pooled when
  `pipe.config.force_zeros_for_empty_prompt` (mirrors diffusers).
- **Schedulers**: a fresh scheduler per job via `from_config(checkpoint scheduler config,
  **overrides)`. Overrides set `use_karras_sigmas` explicitly (False for euler/heun), so a
  checkpoint config can't change the meaning of a sampler.
- **Loading**: `from_single_file` with fp16; sd15 `safety_checker=None,
  requires_safety_checker=False`; sdxl `add_watermarker=False`. img2img = `from_pipe(txt2img)`
  (shared modules, verified by identity in tests); SDXL img2img gets `watermark = None`.
  Model family is explicit; no auto-detect.
- **Optimization profiles**: `baseline` (default until Phase 8), `compile`
  (channels_last UNet + `torch.compile(reduce-overhead, fullgraph)`), `compile-max`
  (+ VAE channels_last and compiled `vae.decode`, max-autotune). Any failure at setup *or*
  in warm-up reverts to the original modules and logs "falling back to baseline". When
  compiled, the first use of a new (mode, w, h) logs "Optimizing new tensor shape…".
- **Generators**: one `torch.Generator(device)` per image. Tests verify an image is the same
  whether generated alone or inside a batch, so OOM splitting doesn't change results.
- **Progress** uses `callback_on_step_end` with `pipeline.num_timesteps` as the total. For
  Heun/DPM2 that's about 2× steps (second-order samplers run more iterations).
- **OOM detection**: `torch.OutOfMemoryError`; on OOM: `gc.collect()` + `cuda.empty_cache()` + log.
- **Allocator**: `configure_cuda_allocator()` sets `PYTORCH_ALLOC_CONF` (setdefault). It only
  works before torch is imported, so the CLI must call it first thing (Phase 4).
- **Warm-up**: txt2img, batch 1, `WARMUP_STEPS = 3` steps, at the family default size
  (overridable by `warmup_size`), output discarded.
- CPU tests lower `MIN_DIMENSION` to 64 via monkeypatch (tiny models at 256 px took 5 min).

### Phase 4 decisions (CLI)

- **Exit codes**: 0 ok; 1 runtime failure (backend load error, generation error, or `doctor`
  found a required capability missing); 2 usage/validation error (argparse also uses 2).
- `--model-family` is required for generate/serve (no auto), optional for doctor. Real mode
  needs `--model` pointing at an existing `.safetensors` file; mock defaults the name to
  `mock.safetensors`.
- Unset knobs come from `FAMILY_DEFAULTS`. `--strength` without `--image` is ignored with a
  note; `--image` without `--strength` uses the family default 0.6.
- `generate`: output paths go to **stdout** (one per line, scriptable); everything else
  (notes, corrections, "Job 1: …" summary, resolved per-image prompts when they differ,
  progress about every 10%, errors) goes to **stderr**. Job id is always 1. Warm-up is
  only run for non-baseline optimization (to trigger the compile fallback).
- `configure_cuda_allocator()` is called in `build_backend` (real) and doctor's GPU checks,
  before torch is first imported.
- **doctor**: `--mock` checks only Python/Flask/Pillow and never imports torch. Real mode
  *fails* on: torch missing, CUDA unavailable, FP16 CUDA alloc failing, diffusers or
  transformers missing, a given checkpoint missing or not `.safetensors`. It *warns* on:
  non-L4 GPU, SDPA/channels_last/torch.compile unavailable, compile smoke failing,
  accelerate/safetensors missing. `--compile-check` runs a tiny `torch.compile` smoke test
  (otherwise "not-run"). Package versions come from `importlib.metadata` (no imports).
- Hidden flags `--device` / `--dtype` (default cuda/float16) exist for debugging.
- `serve` flags: `--host 127.0.0.1`, `--port 8000`, `--inputs-dir`, `--outputs-dir`,
  `--mock-load-seconds 2.0`, `--mock-step-seconds 0.05`, `--optimization`.
- Tests use a fake torch namespace (patched via `cli._import_torch` / `_module_available` /
  `_version`) to simulate L4, other GPUs and no CUDA. Never patch `importlib` globally.

### Phase 5 decisions (server)

- **Threads**: Flask (threaded) + `backend-loader` + `gpu-worker`, all daemons. The loader
  runs `load()` then `warmup()`, then sets state `ready` and the `ready` Event. On failure it
  sets state `error`, stores `backend_error`, marks pending jobs failed, and sets `ready`
  too, so the worker wakes and fails its waiting job. `/api/queue` then returns 503.
- **Worker** takes a job off the queue straight away (even while loading) and makes it
  `current` with status `waiting_for_model`, so `queue_length` counts *pending* jobs only.
  `MAX_QUEUE = 5` pending, so up to 6 jobs exist in total. Full queue → **429**.
- Job statuses: queued, waiting_for_model, running, done, error, cancelled (and "failed" for
  pending jobs drained on load error). `latest_completed_job` changes only when a job
  succeeds completely; on a mid-job error the partial PNGs stay in `outputs/` and are logged.
- **Status JSON** = the SPEC §17 fields + `latest_completed_job.seeds/prompts` +
  `current_job.num_images` + `log` (entries `{seq, text}` after `?log_after=N`; server
  keeps 300 lines). The browser should poll with the last seq it saw. `/api/config` also
  returns `max_queue`.
- **Log lines** (terminal is timestamped, browser gets plain text): server online, backend
  state messages, `Queued job N (waiting for model): K image(s), seeds A–B`, corrections,
  the per-job settings line, per-image prompts when dynamic choices differ, saved files,
  done/errors. Per-step progress goes only in `current_job.message/progress`, not the log.
  Werkzeug request logging is set to WARNING (no line for every poll). The real backend's
  `log` attribute is redirected into the server log.
- **Queue JSON** (`POST /api/queue`): fields as in `GenerationRequest`; missing ones use the
  family defaults; `input_image` is a bare filename in `inputs/` (checked with
  `secure_filename` + exists + extension); `strength` is dropped for txt2img and `null` →
  default for img2img. Response: `job_id, seeds, prompts, corrections, queue_length,
  waiting_for_model`. Errors: 400 validation (JSON `{"error": msg}`), 429 full, 503 backend error.
- Listings: `{"files": [...]}`, newest first by mtime; inputs = png/jpg/jpeg/webp/bmp, outputs = png.
  Symlinks and other files are ignored. `inputs_mtime`/`outputs_mtime` = directory mtimes.
- **Upload**: multipart field `file`, max 50 MB, `secure_filename` (empty → `upload.png`),
  extension allow-list, saved to a hidden `.upload-<uuid>.part` temp file, checked with
  Pillow `verify()`, then renamed to a collision-safe name.
- **Reuse output**: copies the original PNG bytes into `inputs/` (`name_1.png` on collision).
- **Clear-all**: 409 if a job is current or the queue is non-empty (checked under the lock).
  Overwrites with random bytes + fsync, then unlinks every regular file in inputs/ and
  outputs/, *except `.gitkeep`*. Skips symlinks and subdirectories. Resets the latest job.
- Test helper `GatedBackend` (in `test_server.py`) blocks load/generate on Events for
  deterministic thread tests. The server tests passed 15/15 repeated runs.
- **Phase 6 notes**: the UI should use relative URLs (`api/status`, not `/api/status`) so
  it works behind path-prefixed proxies.

### Phase 6 decisions (UI)

- Based on the reference `videomodeltests/web/static/index.html` (cloned to the scratchpad
  for reading only, not vendored): same palette, 520px card, pill, sections, details
  panels, danger zone, `api()` helper, dropdowns that keep their selection, refresh only
  when the folder mtime changes.
- Differences from the reference: **relative URLs** (`api/status`, `outputs/<name>`) for
  proxies; input defaults to **None (text-to-image)**, and the choice is remembered in
  `sessionStorage` (`sdmodeltester.input`) and restored if the file still exists. The log
  comes from the **server** (`log_after` seq) and the client adds only its own errors and
  offline/reconnect lines. Capped at 100 lines. Polling is a `setTimeout` chain (no overlap).
- Pill: Loading/Optimizing/Warming (amber pulse), Ready (green), Generating (blue pulse),
  Error/Offline (red). Progress area: running → "Job N: Batch 1/2 · denoising 14/25" + %,
  waiting/loading → backend message + indeterminate bar, error → backend_error, idle → "Idle".
- Viewer: one large image (`object-fit: contain`, max-height 70vh) wrapped in a link that
  opens the raw PNG in a new tab, plus an "Open PNG" link. Meta "Image i/N · seed S" and
  that image's resolved prompt. Horizontal 64px thumbnail strip; the first image is
  selected automatically when a new job completes (detected by `latest_completed_job.id` changing).
- Previous outputs: 5 newest thumbnails in a 5-column grid, a dropdown of all outputs,
  clicking a thumbnail selects it in the dropdown, and the button calls `api/reuse-output`
  and auto-selects the new input.
- Form: an empty number field sends `null` (server uses the default); strength is disabled
  for text-to-image; Generate stays enabled while loading; label "Generate N images";
  randomize seed uses `crypto.getRandomValues` (0..2³²−1). Prompt max 4000 chars (matches
  the backend).
- Global `[hidden] { display: none !important }` — needed because `.viewer a { display: block }`
  overrode the hidden attribute (found from a screenshot).
- **UI tests** (`tests/test_ui.py`) run a real werkzeug server (`make_server`, port 0) with
  `ServerState` + mock/Gated backend, and use headless Chromium. They cover: config defaults, labels,
  seed, queueing while loading, 10-image strip, img2img upload + session persistence,
  upload/validation errors, previous outputs + reuse, clear queue / clear-all (409, dismiss,
  accept), XSS (prompt rendered as text), offline/reconnect (route abort), backend error,
  no horizontal overflow at 360px. 3 repeat runs passed; an innerHTML bug is caught.
- Exploration/screenshot scripts live in the scratchpad (not the repo).

### Phase 8a decisions (benchmark tooling)

- `python -m src.cli benchmark --model-family F --model M [--profiles baseline,compile]
  [--verify-profile P] [--no-functional|--no-perf] [--steps N --width W --height H]
  [--output-dir outputs/benchmark] [--report compat/benchmark-F-DATE.md] [--in-process] [--mock]`.
- With more than one profile, each runs in a **fresh child process** (`--child-json` internal),
  so compile caches and VRAM never carry over. The parent never imports torch/CUDA; the
  environment rows come from the child's `collect_diagnostics`. A crashed child is recorded
  as `{"error": ...}` and doesn't abort the run.
- Per profile it measures: load, warmup, warm_single_1..3 (median of #2,#3 = the warm
  latency), batch_5, batch_10 (+ images/s), new_size_first/second (ALT_SIZES sd15 768x512,
  sdxl 896x1152), img2img_first/second, peak VRAM (reset before each batch), learned OOM
  batch limits.
- **Recommendation** = lowest warm-single median among profiles that ran *and* stayed active
  (a compile that fell back to baseline doesn't count). A compiled profile must beat baseline
  by ≥5% (`MIN_SPEEDUP`), otherwise baseline. Startup is reported next to it.
- **Functional matrix** (18 checks) on the verify profile's loaded backend: txt2img/img2img
  ×1/×10, 6 samplers, seed −1, same-seed reproducibility (≤2 px diff), batch invariance
  (≤24, informational tolerance for GPU kernels), weighted ≠ plain, dynamic, dynamic+weighted,
  non-square, repeated singles without reload (`load_count`). **Every output image is checked
  for black** (max channel ≤ 8), which catches the SDXL fp16 VAE NaN issue. A failing check is
  recorded, never raised.
- Contact sheet `outputs/benchmark/functional_contact_sheet.png` (one labelled row per check),
  because image quality must be judged by a human. Exit code 1 if any check fails or no
  profile completed.
- The mock image no longer draws the batch index (only family/mode/seed/prompt), so a mock
  image depends only on seed/prompt/mode, like the real backend; batch-invariance works on mock.
- Benchmark images go to `outputs/benchmark/<profile>/` subdirectories: they're not listed by
  the UI and not removed by clear-all (which skips subdirectories).

### Phase 9 prep: reference notebook review (owner-provided)

Reviewed `je4ngomes/fast-stable-diffusion/fast_stable_diffusion_AUTOMATIC1111_py310.ipynb`
(A1111 on Colab with a py3.10 venv, torch 2.1.2, xformers, gradio --share).
- **Reuse (ideas):** URL source classification (civitai / huggingface / gdrive / other);
  Civitai filename from `Content-Disposition` (or the redirect URL's
  `response-content-disposition` query); streamed download with a progress bar; a minimum-size
  sanity check; a "use an existing local path (e.g. on Drive)" alternative; a quiet-on-success,
  full-output-on-failure `run()` helper for install steps; streaming server output so errors
  stay visible.
- **Don't reuse (bugs/risks):** `get_name()` does a non-streaming `requests.get`, so a 200
  response would pull a multi-GB body into RAM; no HF/Civitai token support; downloads go
  straight to the final path (a partial file looks "already exists"; no resume); HF links via
  gdown (a `/blob/` URL downloads HTML); the cleanup `!rm model` deletes a literal "model" file;
  the SD1.5 default link `runwayml/stable-diffusion-v1-5` is dead (use
  `stable-diffusion-v1-5/stable-diffusion-v1-5`); the A1111/venv/xformers/gradio setup conflicts
  with our spec (Colab-owned torch, no xformers, built-in proxy only).
- **Plan:** own helpers (outside `src/`, e.g. `notebooks/colab_utils.py`, unit-tested with a
  local HTTP server): `.part` + Range resume, Bearer token never forwarded to the redirect host,
  `hf_hub_download` for HF URLs (blob/resolve parsing), Civitai model-page →
  `api/download/models/{versionId}` (or the v1 API for the latest version), safetensors header
  validation (8-byte length + JSON) + a warn-only family hint from tensor key names.

### Needs Colab (Phase 8b) — cannot be verified locally

- `from_single_file` on real SD1.5/SDXL checkpoints (and Hub config fetch) in fp16 on CUDA.
- SDXL fp16 VAE decode (NaN/black images): diffusers upcasts via the VAE `force_upcast`
  config; confirm.
- Real `torch.compile` behaviour/speed per profile; CUDA-graph warm-up step count.
- Real OOM thresholds at 1024² for batch 10; `expandable_segments` effect.
- Visual check that weighting/samplers look right on a real model.

## Open questions / TODO
- SPEC/COLAB_COMPATIBILITY still reference Python 3.12 — consider updating docs.
- Colab proxy check (Phase 9): confirm relative URLs and "open PNG in new tab" work through
  `serve_kernel_port_as_iframe`.

## Learnings

- Ruff ≥ 0.16 formats code blocks inside Markdown — exclude docs from ruff.
- Pillow EXIF test: save a JPEG with `exif[0x0112] = 6`; `ImageOps.exif_transpose` rotates it.
- transformers 5's `CLIPTokenizer` is tokenizers-backed: build it offline with
  `CLIPTokenizer(vocab=dict, merges=[])`; `bytes_to_unicode` is in
  `transformers.convert_slow_tokenizer`. A char-level vocab (256 bytes + `</w>` variants +
  specials) is enough for tests. Put EOS as the highest id; config `eos_token_id` must match.
- Diffusers' tiny-pipeline configs (32/64 channels, 5-layer CLIP of width 32) build in <1 s,
  with no network.
- Mutation checks run: a single shared generator → the batch-invariance test fails;
  skipping `_apply_weights` → the weighting tests fail.
- Bug caught by tests: the remainder batch (e.g. the final 1 of 7 at size 3) originally
  overwrote the learned OOM limit with 1. Fixed: only full-size successes update the limit.
- A deliberate "stretch instead of crop" bug makes `test_preprocess.py` fail, so the
  no-distortion tests do catch that regression.
