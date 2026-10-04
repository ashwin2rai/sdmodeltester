# Stable Diffusion Image Test UI — V1 Specification

The single source of truth for what this repo must do. Owner decisions made during
development are folded in and marked **(owner)**. Progress and learnings: `status.md`.

## 1. Purpose

A deliberately small harness for testing **SD 1.5 and SDXL single-file `.safetensors`
checkpoints** on a **Google Colab NVIDIA L4**. One checkpoint is loaded once, kept resident on
the GPU, and driven from a minimal browser UI exposed through Colab's built-in kernel proxy.
The backend exists to enable that loop, not as a general inference service.

Dominant workflow: generate one image → tweak the prompt → repeat. Secondary: 5–10 variations
of one prompt, truly batched. Optimize warm single-image latency first, batch throughput second.
Inspired by `videomodeltests` (https://github.com/ashwin2rai/videomodeltests) but narrower.

## 2. Scope

**In V1:** one `.safetensors` checkpoint per session with an explicit family (`sd15` | `sdxl`);
txt2img and img2img; real Diffusers backend + mock backend; CLI (`doctor`, `generate`, `serve`,
`benchmark`); one GPU worker queue; batching (1–10 images) with OOM fallback; six samplers;
`(text:1.2)` weights; one-level `{a | b}` alternatives; negative prompts; seed −1 / sequential
seeds; minimal UI with input upload and output reuse; metadata-free PNGs; clear queue; best-
effort secure clear-all; Colab notebook (built last); HF + Civitai downloads; built-in proxy only.

**Out:** GGUF, Diffusers folders, SD 2.x/SD3/Flux, model switching or multiple models, LoRA,
textual-inversion UI, ControlNet, inpainting, upscaling, SDXL refiner, auth, WebSockets/SSE,
multiple GPU workers, mid-generation cancel, queue editing, ngrok/Cloudflare, Docker/RunPod,
generation database, sidecar JSON, prompt metadata in PNGs, family auto-detection.

## 3. Platform & dependencies

- **Colab owns the platform**: Python (3.13 as of 2026-09), CUDA, NVIDIA driver, PyTorch. Never
  upgrade/replace torch in normal setup; `install_requirements` warns if pip changed it.
- `requirements.txt`: app deps only (flask, pillow) — the normal test suite needs no torch,
  diffusers, CUDA or model download. `requirements-inference.txt`: diffusers, transformers,
  accelerate, safetensors, huggingface_hub — **no torch**. Version ranges are chosen only after
  real L4 testing (§13); prompt weighting is in-house, so no extra dependency.
- No xFormers, FlashAttention wheels, bitsandbytes or custom CUDA extensions: FP16 + PyTorch
  SDPA is the baseline; optional acceleration is capability-detected and falls back.
- Behaviour prefers capability checks over version checks. Don't persist compiler caches
  across sessions.
- After real L4 testing, record a known-good snapshot in `compat/known-good-colab.md` (date,
  runtime, Python/torch/CUDA/GPU/VRAM, diffusers/transformers/accelerate versions, SDPA and
  compile status, chosen profile, SD1.5 + SDXL numbers). It is a recovery reference, not an
  installer. When Colab changes: run `doctor`, the mock suite, one SD1.5 + one SDXL checkpoint
  through `benchmark`, then update ranges narrowly and the snapshot.

## 4. Model loading & defaults

- Family is explicit and trusted: `sd15` → `StableDiffusionPipeline`, `sdxl` →
  `StableDiffusionXLPipeline`, both via `from_single_file` (Hub config fetch allowed), FP16 on
  CUDA, no CPU offload. Incompatible checkpoints fail with a clear error.
- SD1.5 safety checker disabled; SDXL invisible watermark disabled (txt2img and img2img).
- img2img reuses the loaded components (`from_pipe`); no second copy of the weights.

| | SD 1.5 | SDXL |
|---|---:|---:|
| size | 512×512 | 1024×1024 |
| steps / CFG | 25 / 7.5 | 25 / 5.0 |
| sampler | DPM++ 2M SDE Karras | DPM++ 2M SDE Karras |
| seed / images / strength | −1 / 1 / 0.60 | −1 / 1 / 0.60 |

Hidden in V1: clip skip (None), SDXL second prompts (reuse main), SDXL size conditioning
(pipeline defaults), refiner, offload, VAE slicing/tiling.

## 5. Samplers

App-owned IDs (never Diffusers class names), each built per job from the checkpoint's scheduler
config with explicit overrides, without reloading weights:

| ID | Label | Scheduler |
|---|---|---|
| `dpmpp_2m_karras` | DPM++ 2M Karras | DPMSolverMultistep, `dpmsolver++`, Karras |
| `dpmpp_2m_sde_karras` | DPM++ 2M SDE Karras | DPMSolverMultistep, `sde-dpmsolver++`, Karras |
| `euler` | Euler | EulerDiscrete (Karras off) |
| `euler_a` | Euler a | EulerAncestralDiscrete |
| `heun` | Heun | HeunDiscrete (Karras off) |
| `dpm2_karras` | DPM2 Karras | KDPM2Discrete, Karras |

## 6. Prompts

Order: validate braces → choose alternatives per image → weighted embeddings → pipeline.

- **Dynamic** `a man with {white | black} hair`: one brace level; ≥ 2 non-empty, trimmed
  alternatives; several groups allowed; nested, empty or unmatched braces are errors; works in
  positive and negative prompts. Choices use a PRNG seeded from each image's own seed (XOR a
  fixed salt), never the diffusion generator, so they're reproducible per image **(owner:
  images in a fixed-seed job pick independently)**.
- **Weighted** `(white hair:1.2)`: only explicit numeric weights are syntax; other parentheses
  are literal; `\(`/`\)` escape. Implemented in `src/prompting.py` (A1111 "original" emphasis:
  scale token embeddings, restore each chunk's mean); unweighted prompts match diffusers'
  own encoding. Long prompts use 75-token chunks; a batch with different prompts is padded to
  a common length; identical prompts are encoded once. SDXL uses both encoders' penultimate
  states + pooled output; an empty SDXL negative becomes zeros (diffusers behaviour).

## 7. Requests, seeds, inputs, outputs

- Seeds: −1 → a unique random seed per image, resolved at enqueue; fixed seed `s` → `s, s+1, …`.
  One generator per image, so an image is identical alone or in a batch.
- Limits (backend-enforced): width/height 256–2048, **rounded to the nearest multiple of 8
  first (owner)**, UI step 64; steps 1–150; CFG 0–30; images 1–10; prompt ≤ 4000 chars; seed −1 or
  0..2³²−1 with no wraparound. Corrections are applied silently and logged.
- **img2img steps (owner)**: Diffusers runs `int(steps × strength)` steps; if that is 0, raise
  steps to the minimum giving one step (A1111-like) and log it. Strength in (0, 1], ignored for
  txt2img.
- Input images: EXIF-orient, RGB, resize (LANCZOS) to cover the target, center-crop — never
  stretch. No fit-mode choice.
- Outputs: PNG only, no prompt/parameter metadata, no sidecars. Name
  `20261003_162455_job0007_img01_seed123.png` (collision-safe).

## 8. Backend & performance

- Contract: `load(status)`, `warmup(status)`, `generate(job, output_dir, progress)`; the mock
  and real backends share it and the job loop. Torch/diffusers are imported lazily, only on the
  real path.
- **Batching**: one queued job = one logical result. Use true batches; on CUDA OOM halve the
  micro-batch and retry; remember the size that worked per (mode, width, height) for the
  session. Partial PNGs from a failed job stay on disk but never become the "latest" result.
  Progress via `callback_on_step_end`, e.g. `Batch 1/2 · denoising 14/25`, continuous overall.
- **Mock backend**: no torch/diffusers/network/GPU; same types, seeds, prompts, preprocessing,
  paths; placeholder PNGs showing family/mode/seed/prompt; configurable delays, simulated OOM
  and failures.
- **Optimization profiles**: `baseline` (FP16 + SDPA; default until L4 benchmarks decide),
  `compile` (channels_last UNet + `torch.compile(reduce-overhead)`), `compile-max` (+ compiled
  VAE decode, max-autotune). Any failure — at setup or warm-up — falls back to baseline; never
  fatal. New shapes may recompile (logged); no dynamic-shape compile, nightly torch or patched
  model code.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set before torch is imported.
- Startup: HTTP server first; the model loads, optimizes and warms up (one batch-1 generation at
  the family default size, no output) in the background; the UI is usable and jobs can queue
  meanwhile.

## 9. Server & HTTP API

Flask + one loader thread + one GPU worker. Backend states: starting, loading, optimizing,
warming, ready, error. A job dequeued before readiness shows as `waiting_for_model`. Bounded
queue of 5 pending jobs (429 when full). A fatal load error fails waiting/pending jobs, rejects
new ones (503) and leaves the UI up for diagnosis.

| Endpoint | Behaviour |
|---|---|
| `GET /` | the UI |
| `GET /api/config` | family, model name, family defaults, samplers |
| `GET /api/status` | backend state/message/error, queue length, current job (id, status, message, progress), latest completed job (id, outputs, seeds, prompts), inputs/outputs mtimes, log lines after `?log_after=N` |
| `GET /api/inputs`, `/api/outputs` | newest-first filenames |
| `GET /inputs/<name>`, `/outputs/<name>` | file (confined to the directory) |
| `POST /api/upload` | multipart image, validated by Pillow, collision-safe name |
| `POST /api/reuse-output` | copy an output PNG's bytes into `inputs/` |
| `POST /api/queue` | enqueue (seeds/prompts resolved now) → `{job_id, seeds}`; 400 / 429 / 503 |
| `DELETE /api/queue` | drop pending jobs; never the active one |
| `DELETE /api/clear-all` | 409 if a job is active or queued; else overwrite-then-unlink files in `inputs/` and `outputs/` (best effort — can't defeat SSD wear levelling, snapshots or cloud copies) |

## 10. UI

One self-contained `index.html` (inline CSS/JS, no build step) in the `videomodeltests` style:
dark 520px card, status pill (Loading/Optimizing/Warming/Ready/Generating/Error/Offline), ~1 s
polling, relative URLs (proxy-safe), `textContent` only for user/server strings.

Sections: input image (None = text-to-image by default, upload, thumbnail, remembered for the
session); positive and negative prompts with a syntax hint; knobs (width, height, steps, CFG,
sampler, seed + randomize, images, strength only for img2img); Generate (usable while loading)
with queue count and Clear queue; latest result (one large image that opens the PNG,
horizontal thumbnail strip, seed + resolved prompt); progress bar; generation log (server lines,
last ~100); previous outputs (5 newest thumbnails, dropdown of all, reuse as input); danger
zone (confirm, then clear all).

## 11. CLI

`python -m src.cli`: exit codes 0 ok / 1 runtime failure / 2 usage. `--model-family` is
required (no auto); real mode needs an existing `.safetensors` `--model`; `--mock` works
everywhere without torch.
- `doctor [--mock] [--compile-check]`: Python, torch, CUDA, GPU/VRAM (warn if not L4), FP16 CUDA
  tensor, SDPA, channels_last, torch.compile (+ smoke test), diffusers/transformers/accelerate/
  safetensors, allocator setting, family and checkpoint. Fails only on what the command needs.
- `generate`: unset knobs from family defaults; output paths on stdout, progress on stderr.
- `serve [--host --port --inputs-dir --outputs-dir]`.
- `benchmark [--profiles baseline,compile] [--no-functional]` (§13).

## 12. Colab notebook

**(owner)** A minimal, shareable orchestrator — it clones this public repo and never contains a
second implementation of anything. Four form cells, plain Python, no magics:
1. **Settings** — `MODEL_FAMILY`, `MODEL_URL`, optional `HF_TOKEN` / `CIVITAI_TOKEN`, `PORT`.
2. **Install** — clone to `/content/sdmodeltester` (or `git pull` when rerun), install both
   requirement files on top of Colab's torch, run `doctor` (stop on failure).
3. **Download** — Hugging Face file links via `hf_hub_download`; Civitai version links
   (`?modelVersionId=` or `/api/download/models/<id>`) with the token as `?token=`; stream to a
   `.part` file; the result must be a full-size `.safetensors` file with a valid header.
   Tokens: the value typed in the notebook, else the Colab Secret of the same name; never
   printed, and errors never echo URLs.
4. **Start the UI** — `serve` in the background, UI embedded with
   `serve_kernel_port_as_iframe`; rerunning restarts it.

**(owner)** The notebook is quiet and "deaf" to the UI: one ✓ line per cell, details only on
failure, no prompts or server log (that goes to `server.log`; the UI shows progress and errors).
Helpers live in `notebooks/colab_utils.py`; neither imports `src/`. No tunnels; Colab's resource
policies apply.

## 13. Verification

- **Local (no GPU)**: `pytest` passes without torch/diffusers. Optional layers: the real backend
  on CPU with tiny random pipelines (`--group inference-cpu`), and browser tests of the UI
  (`--group ui`). GPU-only tests carry `@pytest.mark.gpu` and are excluded by default.
- **L4 (`benchmark`)**: per profile — cold load, compile/warm-up, warm single ×3, batch 5/10,
  a new resolution, img2img, peak VRAM, OOM limits. Functional matrix (first profile): txt2img
  and img2img ×1/×10, all six samplers, seed −1, reproducible fixed seeds, batch invariance,
  weighted, dynamic and combined prompts, non-square, no reload between jobs; every image is
  checked for black output. Report `compat/benchmark-<family>-<date>.md` + contact sheet for a
  human quality check. Default profile = fastest warm single image that stayed compiled and
  beats baseline by ≥ 5%; no arbitrary seconds-per-image target.
- **V1 is done when**: the local suite passes; mock `serve` covers every UI flow; the UI works
  during warm-up; real SD1.5 and SDXL checkpoints run txt2img + img2img on an L4 without
  reloading; 10-image jobs batch and fall back on OOM; seeds, weights, dynamic prompts and all
  six samplers work for both families; inputs are never distorted; the viewer, previous-outputs
  and clear-queue/clear-all behave as above; PNGs carry no generation metadata; safety checker
  and watermark are off; the notebook never reinstalls torch and exposes the UI only through
  the built-in proxy; optional acceleration falls back safely; a known-good snapshot and
  benchmark results are committed.

## 14. Security & logging

No app auth (personal tool behind Colab's proxy). Enforce: `secure_filename`, no arbitrary
paths over HTTP, names resolved only inside `inputs/`/`outputs/`, upload type checks, model path
only at startup, tokens never sent to the browser or logged. The server log records state
changes, jobs (settings, seeds, resolved prompts), OOM fallbacks, saved files and errors with
tracebacks in the terminal; the browser shows a short subset.

## 15. Later (V2 candidates)

GGUF or Diffusers-folder backends, per-model profiles from Civitai metadata, LoRA, ControlNet,
inpainting, model switching, nested dynamic prompts, history/metadata opt-in, Docker/RunPod,
tunnels, a full-tab Colab proxy once Google stabilizes one. Public interfaces (request fields,
sampler IDs, CLI flags, HTTP JSON) stay independent of Diffusers so backends can change.

## 16. References

Diffusers: single-file loading, `torch2.0` optimization, fp16 acceleration, batched inference,
schedulers (incl. the A1111 mapping), weighted prompts, callbacks, pipeline `from_pipe`, SDXL
watermark — https://huggingface.co/docs/diffusers · PyTorch CUDA allocator —
https://docs.pytorch.org/docs/stable/notes/cuda · Colab runtime versions and FAQ —
https://research.google.com/colaboratory/faq.html · Colab kernel-port helpers —
https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py · HF Hub
downloads — https://huggingface.co/docs/huggingface_hub/guides/download · Civitai downloads —
https://github.com/civitai/civitai-developer-docs · NVIDIA L4 —
https://www.nvidia.com/en-us/data-center/l4/
