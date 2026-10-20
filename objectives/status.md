# Development Status

Persistent memory for development sessions — read first when resuming, update at every pause.
What the repo must do (incl. all owner decisions) is in `SPEC.md`; this file is how we work,
where we are, and what we learned.

**Last updated:** 2026-10-17

## Working agreement (owner)

- Memory lives here, not in temp folders or agent-private memory.
- Never commit or push; stop at natural pauses so the owner can review, test and commit.
- Small steps. Less code is better while meeting the spec.
- Small Codespace (2 CPU, ~8 GB, no GPU): no large data downloads (checkpoints, big images —
  generate test images with Pillow). Libraries and tools are fine to install without asking
  (never the multi-GB CUDA torch build). Use uv.
- Build and test everything possible with mocks first; real-GPU work needs Colab.

## Environment

- Python 3.13 (matches Colab). `uv sync`, then `uv run pytest`, `uv run ruff check .`,
  `uv run ruff format --check .`. Everything: `uv run --all-groups pytest` (~75 s).
- Optional groups: `inference-cpu` (CPU torch from the `pytorch-cpu` index + diffusers stack, for
  tiny-model tests) and `ui` (playwright; `playwright install chromium --only-shell`, system libs
  once with `sudo .venv/bin/playwright install-deps chromium-headless-shell`).
- uv syncs exactly: a plain `uv run` removes optional groups. To run the torch-free suite without
  touching `.venv`: `UV_PROJECT_ENVIRONMENT=<scratch>/venv-notorch uv run pytest`.
- Ruff skips `objectives/` and `*.md` (it formats Markdown code blocks); notebooks: E501 off,
  not formatted.

## Progress

| Step | State |
|---|---|
| Core logic, mock + real backends, CLI, server, UI, local verification (SPEC §4–11, §13 local) | done |
| `benchmark` tooling (SPEC §13) | done |
| Colab notebook + helpers (SPEC §12), consolidation, notebook tokens | done — awaiting owner review/commit |
| Run `benchmark` on a real L4 for SD1.5 + SDXL, pick the default profile, write `compat/known-good-colab.md` | needs Colab |
| Run the notebook on Colab, fix what reality finds, pin `requirements-inference.txt` ranges | needs Colab |
| Z-Image Turbo family (`zimage`, SPEC §4a): BF16 + plain FP8 transformer files, tiny-model CPU tests through the real single-file loader | done locally — awaiting owner review; needs Colab |
| Fix: img2img `from_pipe` upcast SD/SDXL to FP32 (SPEC §4) | done — re-measure SD/SDXL speed/VRAM on Colab |

## Verified against Colab without a GPU (2026-10-04)

- Colab's GPU image (`googlecolab/backend-info` `pip-freeze.gpu.txt`, updated 2026-10-02):
  Python 3.13.15, torch 2.11.0+cu130, diffusers 0.40.0, transformers 5.18.0, accelerate 1.15.0,
  safetensors 0.8.0, huggingface_hub 1.33.0, Flask 3.1.3, Werkzeug 3.1.9, pillow 11.3.0,
  numpy 2.1.3 — everything we need is preinstalled.
- `uv pip install --dry-run -r requirements.txt -r requirements-inference.txt` on top of exactly
  those versions: **"Would make no changes"** (cell 2 cannot touch torch).
- Full suite in a venv with those exact versions (torch 2.11.0 CPU build): 251 passed (UI tests
  skipped there; no playwright).
- Live endpoints, headers only: HF `stable-diffusion-v1-5/stable-diffusion-v1-5/
  v1-5-pruned-emaonly.safetensors` (3.97 GiB) and `stabilityai/stable-diffusion-xl-base-1.0/
  sd_xl_base_1.0.safetensors` (6.46 GiB) exist; Civitai (Juggernaut XL, version 1759168): both
  the page URL and the download link → one 307 to Cloudflare R2 → 200 octet-stream with
  Content-Length; our filename logic gives `juggernautXL_ragnarok.safetensors`.
- To refresh this check later: download `pip-freeze.gpu.txt`, build a venv with those pins
  (CPU torch from the pytorch-cpu index), dry-run the requirements, run pytest.

## First real Colab L4 run (2026-10-04, owner)

- SD1.5 community checkpoint (`realismByStableYogi_sd15V9.safetensors`): `from_single_file`
  FP16 on CUDA loaded in 11.5 s, warm-up 1.3 s, backend ready at 12.8 s. Diffusers logs a
  harmless "modules … should be kept in float32: []" notice (empty list).
- **The iframe did not appear — root cause: third-party cookies blocked** (Chrome incognito
  blocks them by default). Symptoms: iframe silently not drawn, full-tab `proxyPort` URL 404 on
  every path, `cache_in_notebook=True` iframe read-only ("500 Not allowed" on POST/DELETE).
  Saving a copy to Drive alone didn't fix it; allowing third-party cookies did (accessAllowed was
  True). The intro now tells users to allow `[*.]colab.dev` / `[*.]googleusercontent.com`.
- Owner preference: a UI in its own tab → optional cell 5 prints the `proxyPort` link with its
  limitations; full-tab behaviour with cookies allowed is not yet confirmed.
- Cell 2 now `git pull`s when rerun, so pushing a fix + rerunning cells 2 and 4 updates a live
  session.

## Colab image drift (2026-10-20)

- Colab now preinstalls diffusers 0.41.0 + transformers 5.20.1 (the published
  `pip-freeze.gpu.txt` still says 0.40.0 / 5.18.0) alongside torchao 0.10.0. diffusers 0.41.0
  imports `torchao.quantization.FqnToConfig` (newer torchao only) whenever torchao is present,
  so `doctor` failed on the Z-Image pipeline import. Fix: `diffusers>=0.36,<0.41` (reproduced
  and verified on CPU with torchao 0.10.0). Lift the cap once Colab's torchao has `FqnToConfig`.
- `doctor` now reports the innermost exception (diffusers wraps import errors).

## Still unverified (needs Colab)

- Z-Image on the L4 (2026-10-17, built without downloading the 7.7 GiB text encoder): real
  `from_single_file` on the Comfy BF16 and drbaph FP8 files (headers were checked remotely:
  same 453 tensor names, plain casts), Hub fetch of `Tongyi-MAI/Z-Image-Turbo` components,
  VRAM (est. BF16 ~19.3 GiB / FP8 ~13.5 GiB resident) and batch OOM limits at 1024², speed
  (est. ~1 s/step), FP8 vs BF16 quality (e5m2 expected worse), `compile` on the transformer.
- SD/SDXL timings and VRAM from the 2026-10-04 run were measured while everything ran in FP32
  (`from_pipe` bug); they should drop after the fix.

- `from_single_file` FP16 on CUDA for both families (Hub config fetch); SDXL FP16 VAE black
  images; `torch.compile` speed per profile; OOM thresholds at 1024² × 10; visual quality of
  weighting/samplers.
- The iframe proxy loads the UI; relative URLs and "Open PNG" work through it; `pip install`
  leaves Colab's torch alone; real HF (gated) and Civitai (`?token=`) downloads and naming.
- On Colab, run the benchmark from a terminal or scratch cell in `/content/sdmodeltester` after
  stopping the UI server (it needs the VRAM).

## Implementation notes

- Code map: `src/prompting.py` (dynamic + weighted prompts), `src/backend.py` (types, request
  normalization/validation, `MicroBatcher`, `_BatchedBackend` job loop, mock + Diffusers
  backends), `src/server.py`, `src/cli.py`, `src/benchmark.py`, `src/static/index.html`,
  `notebooks/colab_utils.py` + `colab.ipynb`.
- `MicroBatcher` stores a size limit only after an OOM, and only from a full-size success (a
  smaller remainder batch never lowers it).
- `benchmark` runs profiles one after another in one process with a fresh backend each; Inductor
  caches persist, so for clean compile timings run one profile per invocation.
- `colab_utils.stop_server` reaps its own child (a zombie still "exists" for `killpg(pid, 0)`).
- The notebook JSON was generated; edit a cell's `source` lines and rerun
  `tests/test_notebook.py` (it runs the real cells against a temporary git repo, a local model
  host and a real `serve`, and asserts the exact quiet output).
- Test helpers: `make_request()` in `tests/test_seeds.py`; `GatedBackend` in
  `tests/test_server.py` (load/generate block on Events for deterministic thread tests); tiny
  SD1.5/SDXL pipelines + an offline CLIP tokenizer in `tests/test_diffusers_cpu.py`.

## Learnings

- Civitai "fp8" Z-Image fine-tunes (e.g. moodyProMix) are ComfyUI scaled FP8: `weight` F8 +
  `weight_scale` F32 + `comfy_quant` U8 per linear layer, under `model.diffusion_model.`. Our
  first header check rejected them; they are now dequantized at load (2026-10-17).

- diffusers 0.40 `from_pipe` casts all shared modules to `dtype` (default float32) — pass the
  pipeline's dtype; for FP8 layerwise-cast modules build the pipeline from `pipe.components`
  instead (any `.to(dtype)` turns FP8 storage back into BF16). Device-only `.to()` is safe.
- Z-Image tiny pipeline: Qwen3 tokenizer = `PreTrainedTokenizerFast` over a byte-level BPE with
  empty merges + a chat template; transformer `axes_lens[0]` must exceed caption tokens + 1
  (prompts are padded to 512, real model uses 1024). Comfy layout = fused `attention.qkv`,
  `out`, `k_norm`/`q_norm`, `x_embedder.`/`final_layer.` (inverse of diffusers' converter).

- transformers 5 `CLIPTokenizer` is tokenizers-backed: build offline with
  `CLIPTokenizer(vocab=dict, merges=[])`; `bytes_to_unicode` is in
  `transformers.convert_slow_tokenizer`; EOS must be the highest id.
- Tiny Diffusers pipelines build from configs in < 1 s without network; CPU tests lower
  `MIN_DIMENSION` to 64 (256 px took minutes).
- Flask lambda routes all get endpoint `<lambda>`; name endpoints explicitly and don't reuse view
  names.
- `requests` exception messages contain the URL — never let them reach the notebook when a token
  is in the query string.
- Code review (2026-10-04) found and we fixed: a pasted Civitai `?token=` was dropped; non-ASCII
  upload names were rejected (`secure_filename` strips them → `upload.<ext>` fallback); the worker
  dequeued outside the lock (clear-all could race a job — dequeue + `current` now atomic);
  clear-all could delete an in-progress upload (it now skips hidden files, incl. `.gitkeep`);
  scheme-less links were rejected; HF errors were raw tracebacks; a functional-check crash wiped
  a profile's timings; no `empty_cache()` between benchmark profiles.
- Bugs the tests caught: remainder batch overwrote the OOM limit; server stop waited 15 s
  (zombie); benchmark paths relative to the wrong cwd; CSS overriding `[hidden]`.
- Mutations the suite catches: shared generator, skipped weighting, stretch instead of crop,
  `innerHTML` for prompts, `import torch` on a mock path, a token in an error message.
