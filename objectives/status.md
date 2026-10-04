# Development Status & Learnings

Persistent memory for development sessions — read it first when resuming, update it at every
pause. Source-of-truth specs: `SPEC.md`, `COLAB_COMPATIBILITY.md` (owner decisions below
override them where noted).

**Last updated:** 2026-10-04

---

## Working agreement (from the repo owner)

- **Memory lives here**, not in temp folders or agent-private memory.
- **Never commit or push.** Stop at natural pause points; the owner reviews, tests, commits.
- **Small steps** with pauses — don't one-shot the project.
- **Small Codespace** (2 CPU, ~8 GB RAM, no GPU): don't download large *data* (no checkpoints,
  no big test images; generate test images with Pillow). **Libraries/tools are fine** —
  install freely, make the best decision (never the multi-GB CUDA torch build).
- **Use uv.** **Less code is better** while still meeting the requirements.
- **Priority:** everything testable in mock / representative tests first; real-GPU work needs Colab.

## Environment

- Colab defaults to **Python 3.13** (3.13.15 since the 2026-09-16 release; also Ubuntu 24.04,
  tokenizers 0.23.1). Local dev pinned to 3.13 (`.python-version`); `requires-python >=3.10`,
  ruff `target-version = "py310"`. SPEC/COLAB_COMPATIBILITY still say 3.12 (outdated).
- `uv sync` → `uv run pytest` / `uv run ruff check .` / `uv run ruff format --check .`.
  Everything: `uv run --all-groups pytest` (~75 s, 262 tests). Without optional groups: 201
  pass, 3 modules skip. uv syncs exactly, so a plain `uv run` uninstalls optional groups; to
  run the torch-free suite without disturbing `.venv`:
  `UV_PROJECT_ENVIRONMENT=<scratch>/venv-notorch uv run pytest`.
- Optional groups: `inference-cpu` (torch from the `pytorch-cpu` index via `[tool.uv.sources]`,
  diffusers, transformers…) for tiny-model tests; `ui` (playwright; Chromium headless shell via
  `playwright install chromium --only-shell`, system libs once via
  `sudo .venv/bin/playwright install-deps chromium-headless-shell`).
- App deps (flask, pillow ≥10.1) in `[project]` and `requirements.txt`; diffusers stack (no torch)
  in `requirements-inference.txt`; `requests` in dev (for colab_utils tests). No
  `[build-system]`; pytest uses `pythonpath = ["."]`. Ruff excludes `objectives/` and `*.md`;
  notebooks: E501 ignored, not formatted.

## Phase progress

| Step | Description | State |
|---|---|---|
| 0–7 | Setup, core logic, mock backend, real backend, CLI, server, UI, local verification | committed |
| 8a | `benchmark` CLI (profiles + functional matrix + report) | committed |
| 9a | Colab notebook + `notebooks/colab_utils.py` | committed |
| C | **Consolidation**: minimal notebook, smaller utils/CLI/benchmark, shared backend job loop | **done — awaiting owner review/commit** |
| 8b | Run `benchmark` on a real L4 (SD1.5 + SDXL), pick the default profile, write `compat/known-good-colab.md` | needs Colab |
| 9b | Run the notebook on Colab, fix what reality finds, pin `requirements-inference.txt` ranges | needs Colab |

## Code map

- `src/prompting.py` — dynamic `{a | b}` templates (`parse_template`, `resolve_prompt_pair`,
  per-image PRNG = seed XOR salt) and weighted `(text:w)` prompts (`parse_weighted`,
  `chunk_tokens`, lazy-torch `encode_prompt_batch`).
- `src/backend.py` — constants (families, samplers, `SAMPLER_SCHEDULERS`, `FAMILY_DEFAULTS`),
  request types, `normalize_request` → `validate_request` → `resolve_job`, `describe_request`
  (log summary shared by CLI and server), `preprocess_image`, filenames, `MicroBatcher`,
  `_BatchedBackend` (shared job loop: preprocess → micro-batch with OOM fallback → `save_png`;
  subclasses implement `_render`, `_is_oom`, `_on_oom`), `MockBackend`, `DiffusersBackend`.
- `src/server.py` — `ServerState` (queue, loader + worker threads, log ring), `create_app`, `serve`.
- `src/cli.py` — `doctor`, `generate`, `serve`, `benchmark`; `build_backend`, `build_request`,
  `collect_diagnostics`.
- `src/benchmark.py` — `Runner`, `benchmark_profile`, `recommend`, `functional_checks`,
  `contact_sheet`, `render_report`.
- `src/static/index.html` — the whole UI (inline CSS/JS).
- `notebooks/colab.ipynb` + `notebooks/colab_utils.py` — the Colab orchestrator.

## Decisions

### Requests, prompts, seeds
- `resolve_job` = `normalize_request` (silent corrections as `job.corrections` log notes) →
  `validate_request` → seeds → dynamic prompts. Done at enqueue time.
- **Dimensions** (owner): round to nearest multiple of 8 (ties up), then hard range 256–2048
  (rejected, not clamped). UI step 64.
- Limits: steps 1–150, CFG 0–30, images 1–10, prompt ≤ 4000 chars, seed −1 or 0..2³²−1 and
  `seed+n−1 ≤ 2³²−1` (no wraparound).
- **img2img steps** (owner, A1111-like): if `int(steps × strength)` would be 0, quietly raise
  steps to the minimum giving 1 denoising step (5 @ 0.1 → 10) with a note; reject if that needs
  >150. Strength in (0, 1]; `None` for txt2img (server/CLI drop it).
- Seed −1 → unique `SystemRandom` seeds per image; fixed seed → base, base+1, …
- **Dynamic prompts** (owner-confirmed): one PRNG per image from its own seed (positive then
  negative), so an image's picks depend only on its seed. One brace level; `{x}`/empty/nested/
  stray braces are errors; `|` outside braces is literal.
- **Weighting** in-house (A1111 "original": scale by weight, restore chunk mean); unweighted
  prompts exactly match diffusers' `encode_prompt` (tested). Only `(text:number)` is syntax;
  `\(`/`\)` escape; `(ratio:16:9)` = "ratio:16" weight 9 (as A1111). 75-token chunks, all
  prompts in a batch padded to the same count, each distinct string encoded once. SDXL empty
  negative → zeros when `force_zeros_for_empty_prompt`.
- Filenames `YYYYMMDD_HHMMSS_job0007_img01_seed123.png` (one timestamp per job; `_1` on
  collision). PNGs carry no metadata (`save_png` copies pixels into a fresh image).
- Preprocess: EXIF transpose, RGB, cover-resize (LANCZOS), center crop.

### Backends
- `MicroBatcher`: tries the whole job (capped by a learned limit per (mode, w, h)); OOM halves;
  a limit is stored only after an OOM and only from a *full-size* success. Progress
  "Batch 1/2 · denoising 14/25" (no prefix for one batch), continuous across batches.
- `_BatchedBackend.generate` wraps every failure in `GenerationError(msg, completed_paths)`;
  partial PNGs stay. The mock's `_render` is a generator so `fail_after_images` leaves earlier
  images written. Mock knobs: `load_seconds`, `step_seconds`, `max_batch`, `fail_load`,
  `fail_after_images`; mock image = seed colour (or tinted input) + family/mode/seed/prompt text
  (no batch index, so it depends only on seed/prompt/mode like the real backend).
- Real: `from_single_file` fp16; sd15 no safety checker; sdxl `add_watermarker=False` and
  img2img `watermark=None`; img2img via `from_pipe` (shared modules). Fresh scheduler per job
  from the checkpoint config with explicit overrides (`use_karras_sigmas` set explicitly). One
  `torch.Generator` per image (batch-invariant, tested). Progress total = `pipeline.num_timesteps`
  (≈2× steps for Heun/DPM2). OOM = `torch.OutOfMemoryError` → gc + `empty_cache` + log.
- Optimization profiles: `baseline` (default until 8b), `compile` (channels_last UNet +
  `torch.compile(reduce-overhead, fullgraph)`), `compile-max` (+ VAE decode, max-autotune).
  Failure at setup or warm-up reverts to the original modules. Warm-up = 3-step txt2img batch 1
  at the family default size. `configure_cuda_allocator()` (`PYTORCH_ALLOC_CONF=
  expandable_segments:True`) runs before torch is imported. `load_count`, `memory_stats()`,
  `reset_peak_memory()` exist for the benchmark.

### CLI
- Exit codes 0 ok / 1 runtime failure / 2 usage. `--model-family` required (optional for doctor);
  real mode needs an existing `.safetensors` `--model`. Unset knobs from family defaults.
  `generate`: output paths on stdout, everything else (notes, `describe_request` lines, ~10%
  progress) on stderr.
- `doctor`: `--mock` never imports torch. Real mode fails on: torch/CUDA/FP16 tensor missing,
  diffusers/transformers missing, bad checkpoint; warns on non-L4, SDPA/channels_last/compile
  issues, accelerate/safetensors missing. Probes report "ok"/"failed (why)"; `--compile-check`
  runs a tiny `torch.compile` (else "not-run").
- `benchmark --profiles a,b [--no-functional]`: profiles run **in one process, one after another,
  each with a fresh backend** (gc between). The first profile also runs the 18 functional
  checks. Report → `compat/benchmark-<family>-<date>.md` (path printed on stdout), images/contact
  sheet → `outputs/benchmark/`. Caveat: Inductor caches persist within the process, so a later
  compiled profile may show a shorter compile time — run one profile per invocation for clean
  compile timings.
- Benchmark metrics: load, warm-up, warm #1–3 (median of #2/#3 is the warm latency), batch 5/10,
  new size first/second (sd15 768x512, sdxl 896x1152), img2img first/second, peak VRAM during
  batches, OOM limits. Recommendation: fastest warm single among profiles that stayed active; a
  compiled profile must beat baseline by ≥5%. Every functional-check image is tested for black
  (max channel ≤ 8 → catches SDXL fp16 VAE NaNs).

### Server / API
- Threads: Flask (threaded) + loader + worker. Jobs can be queued while loading; the worker takes
  one immediately as `current` (`waiting_for_model`), `queue_length` counts pending only,
  `MAX_QUEUE = 5` → 429 when full. Load failure → state `error`, pending drained, waiting job
  failed, `/api/queue` → 503; the UI stays up.
- `/api/config` = family, model name, `defaults` (= `FAMILY_DEFAULTS` as a dict), samplers.
  `/api/status` = SPEC §17 fields + `latest_completed_job.seeds/prompts` +
  `current_job.num_images` + `log` entries after `?log_after=N` (300-line ring).
  `POST /api/queue` → `{job_id, seeds}`; 400 / 429 / 503 errors as `{"error": msg}`.
- Files: names checked with `secure_filename` + extension + exists (no symlinks); uploads ≤ 50 MB
  verified by Pillow via a hidden `.part` temp; reuse-output copies the PNG bytes; clear-all
  (409 while busy) overwrites + fsyncs + unlinks regular files except `.gitkeep`.
- Werkzeug request logging is off (WARNING); the real backend's `log` goes into the server log.

### UI
- Based on the reference `videomodeltests` UI (palette, 520px card, pill, sections, details
  panels, danger zone). Relative URLs (proxy-safe); input defaults to None and is remembered in
  `sessionStorage`; log comes from the server (`log_after`); `setTimeout` polling; all text via
  `textContent`; global `[hidden]{display:none!important}`. Browser tests in `tests/test_ui.py`.

### Colab notebook (owner decisions)
- **Minimal orchestrator, 4 code cells, no options beyond the spec's inputs**: 1 Settings
  (`MODEL_FAMILY`, `MODEL_URL`, `HF_TOKEN`, `CIVITAI_TOKEN`, `PORT`), 2 Install (clone
  `https://github.com/ashwin2rai/sdmodeltester` to `/content/sdmodeltester` if missing,
  `install_requirements`, `doctor` via the quiet `cu.run`), 3 Download (`fetch_checkpoint` into
  `REPO_DIR/models`), 4 Start the UI (`start_server` + `show_ui`). No demo mode, benchmark cell,
  Drive option, zip/stop cells, or preflight duplicate of `doctor` (owner: absolute minimum).
- **Quiet / "deaf" to the UI**: one ✓ line per cell, details only on failure, never prompts or
  server log (that goes to `/content/sdmodeltester/server.log`). Overrides COLAB_COMPATIBILITY
  §11F (tail logs into the notebook) — the UI shows load progress/errors itself.
- **Decoupled**: the notebook and `colab_utils` never import `src/`; `src/` has no Colab code.
  Interfaces: CLI flags/exit codes and the requirements files.
- `colab_utils` (~240 lines): `get_secret` (Colab Secrets → form → env), `run` (quiet unless it
  fails), `install_requirements` (warns if pip changed torch), `fetch_checkpoint` (HF via
  `hf_hub_download(local_dir)`; Civitai needs the version — `?modelVersionId=` → 
  `/api/download/models/{id}?type=Model&format=SafeTensor`, or the download link itself; direct
  URLs), `download` (requests stream to `.part`, rename when complete, one `\r` progress line,
  Bearer token dropped by requests on cross-host redirects — tested, 401/403/HTML errors),
  `validate_checkpoint` (.safetensors, ≥ 500 MiB, parsable header), `start_server` /
  `stop_server` (pid file, own process group, reaps the child — a zombie looks alive to
  `killpg(pid, 0)`), `show_ui` (`serve_kernel_port_as_iframe`).
- Dropped in consolidation (re-add only if needed): download resume, SHA256 check, Civitai API
  lookup (model page without version), family-mismatch warnings, local/Drive paths.
- Benchmark on Colab: run `python -m src.cli benchmark …` from a terminal/scratch cell in
  `/content/sdmodeltester` after stopping the UI server (VRAM).
- Editing the notebook: generated JSON; change a cell's `source` lines and rerun
  `tests/test_notebook.py` (its end-to-end test runs the real cells against a temporary git
  repo, a local HTTP "model host", and a real `serve`, and asserts the exact quiet output).

## Needs Colab
- Real `from_single_file` (fp16, CUDA, Hub config fetch) for SD1.5 and SDXL; SDXL fp16 VAE
  (black images?); `torch.compile` speed per profile; OOM thresholds at 1024² × 10;
  visual check of weighting/samplers.
- Notebook: iframe proxy loads the UI; relative URLs + "Open PNG" work; `pip install` leaves
  Colab's torch alone; real HF (incl. gated) and Civitai (token) downloads and naming.

## Learnings
- Ruff ≥ 0.16 formats Markdown code blocks — exclude docs.
- transformers 5 `CLIPTokenizer` is tokenizers-backed: build offline with
  `CLIPTokenizer(vocab=dict, merges=[])` (`bytes_to_unicode` in
  `transformers.convert_slow_tokenizer`); EOS must be the highest id.
- Diffusers tiny pipelines build from configs in <1 s with no network; CPU tests lower
  `MIN_DIMENSION` to 64 (256 px took minutes).
- Flask: two lambda routes both get endpoint `<lambda>`; name endpoints explicitly (and avoid
  clashing with existing view names like `inputs`).
- Bugs caught by tests: remainder batch overwrote the OOM limit; stopping a server waited 15 s
  (zombie); benchmark paths relative to the wrong cwd; `[hidden]` overridden by CSS.
- Mutation checks that the suite catches: shared generator, skipped weighting, stretch instead of
  crop, `innerHTML` for prompts, `import torch` in a mock-path module.
