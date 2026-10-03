# Stable Diffusion Image Test UI — V1 Implementation Specification

**Date:** 2026-10-03  
**Status:** implementation-ready specification  
**Reference project:** `videomodeltests`  
**Target:** Google Colab + NVIDIA L4, with local mock-only development on non-GPU machines

---

## 1. Objective

Build a minimal repository for testing **Stable Diffusion 1.5 and SDXL image checkpoints** through a small browser UI, with the real execution environment centered on a Google Colab notebook using an NVIDIA L4.

The repository contains a backend because a persistent process is the simplest way to:

- load a large checkpoint only once;
- keep model components resident in GPU memory;
- queue repeated generations safely on one GPU;
- expose progress and logs to a browser;
- reuse the exact same generation implementation from both CLI and UI;
- support real batch inference for 1–10 output images.

The backend is therefore an **enabling layer for the Colab notebook and UI**, not an attempt to build a general hosted inference platform.

The dominant user workflow is:

```text
one model loaded once
    -> prompt
    -> generate one image
    -> change prompt
    -> generate one image
    -> repeat many times
```

The secondary workflow is:

```text
one prompt
    -> request 5–10 variations
    -> batch as much as L4 VRAM permits
```

The design should optimize warm single-image latency first and batch throughput second.

---

## 2. V1 scope

### 2.1 Required

- Single local `.safetensors` checkpoint.
- Explicit model family: `sd15` or `sdxl`.
- Text-to-image.
- Image-to-image.
- Persistent real backend using Hugging Face Diffusers + PyTorch CUDA.
- Persistent mock backend requiring no Torch/Diffusers/CUDA.
- CLI direct-generation command.
- CLI persistent server command.
- CLI environment diagnostic (`doctor`).
- One GPU worker queue.
- Real multi-image batching with OOM micro-batch fallback.
- Model-family-aware defaults.
- Six curated sampler choices.
- Explicit prompt weighting syntax `(text:1.2)`.
- One-level dynamic prompt alternatives `{one | two | three}`.
- Negative prompts.
- Seed `-1` randomization and deterministic seed sequences.
- Minimal dark browser UI derived from `videomodeltests` interaction patterns.
- Input upload and previous-output-to-input reuse.
- PNG output with no generation prompt metadata embedded.
- Clear pending queue.
- Best-effort secure deletion of all input/output files.
- Colab notebook built as the **final development phase**.
- Hugging Face and Civitai model download helpers in notebook cells.
- Built-in Colab kernel proxy only; no external tunnel in V1.

### 2.2 Explicitly out of scope

- GGUF.
- Diffusers multi-folder model directories.
- SD 2.x.
- SD3, Flux, Illustrious-specific custom architecture support, or other model families as first-class modes.
- Runtime model switching.
- Loading more than one checkpoint per session.
- LoRA loading.
- textual inversion management as a UI feature.
- ControlNet.
- inpainting/masks.
- upscaling.
- SDXL refiner pipeline.
- authentication/user accounts.
- WebSockets/SSE.
- multiple concurrent GPU inference workers.
- mid-generation cancellation.
- per-item queue editing/reordering.
- Cloudflare/ngrok.
- Docker/RunPod deployment in V1.
- persistent generation database.
- prompt/parameters sidecar JSON.
- prompt text embedded in PNG metadata.
- model-family auto-detection.

---

## 3. Design inheritance from `videomodeltests`

Retain these successful patterns:

- thin CLI over a reusable backend;
- one stateful backend instance in server mode;
- a bounded FIFO queue;
- one worker consuming the queue serially;
- polling `/api/status` approximately every second;
- progress callback from inference into server state;
- one self-contained `index.html` with inline CSS/JS and no build step;
- exposed generation log;
- input and output directories;
- file upload validation;
- clear-pending-queue behavior that never interrupts the active generation;
- best-effort overwrite-before-unlink clear-all behavior;
- status pill + progress bar;
- mock backend used by the same API/UI.

Change these patterns where the image use case requires it:

1. **Do not block Flask startup on model loading.** The browser UI must become available immediately while model load/compile/warm-up runs in a background loader thread.
2. **Allow enqueue before readiness.** Jobs submitted while loading are accepted and display `Waiting for model…` until readiness.
3. **Use real tensor batching within a job.** A single queued job may contain up to 10 outputs.
4. **Keep the full Stable Diffusion model on GPU.** No deliberate per-stage CPU unloading in the normal L4 fast path.
5. Replace the video player/frame extractor with an image viewer, thumbnail strip, and previous-output reuse panel.

---

## 4. Proposed repository layout

Keep source count low.

```text
repo/
├── README.md
├── requirements.txt
├── requirements-inference.txt
├── pyproject.toml                    # pytest/ruff config; optional package metadata
│
├── src/
│   ├── __init__.py
│   ├── backend.py                    # request/result types + mock + real Diffusers backend
│   ├── prompting.py                  # brace expansion + weighted embedding adapter
│   ├── server.py                     # Flask, queue, loader/worker threads, filesystem API
│   ├── cli.py                        # doctor / generate / serve
│   └── static/
│       └── index.html                # entire UI, inline CSS/JS
│
├── tests/
│   ├── test_prompting.py
│   ├── test_seeds.py
│   ├── test_preprocess.py
│   ├── test_mock_backend.py
│   ├── test_cli.py
│   └── test_server.py
│
├── models/
│   └── .gitkeep
├── inputs/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
│
├── compat/
│   └── known-good-colab.md           # written/updated only after real L4 verification
│
└── notebooks/
    └── colab.ipynb                   # implemented LAST
```

Do not add a generic `utils.py` unless a genuinely shared abstraction emerges. Prefer explicit small functions in the module that owns the behavior.

---

## 5. Backend application contract

### 5.1 Model family

Exactly:

```python
ModelFamily = Literal["sd15", "sdxl"]
```

No `auto` enum value.

Both CLI and notebook require the user to state the family. The backend trusts that value and selects the corresponding pipeline classes. If the checkpoint is incompatible, loading fails with a clear error.

### 5.2 Generation request

Conceptual request type:

```python
@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    sampler: str
    seed: int                 # -1 or >= 0 on incoming request
    num_images: int           # 1..10
    input_image: Path | None
    strength: float | None    # required only when input_image exists
```

The server converts this into a resolved internal job before the GPU worker runs:

```python
@dataclass(frozen=True)
class ResolvedImageSpec:
    index: int
    seed: int
    prompt: str
    negative_prompt: str

@dataclass(frozen=True)
class ResolvedGenerationJob:
    request: GenerationRequest
    images: tuple[ResolvedImageSpec, ...]
```

### 5.3 Result

Conceptual result:

```python
@dataclass(frozen=True)
class GenerationResult:
    output_paths: tuple[Path, ...]
    seeds: tuple[int, ...]
    resolved_prompts: tuple[str, ...]
    elapsed_seconds: float
```

Resolved prompts/seeds may be returned to the UI/log for the live session but must **not** be embedded into output PNG metadata.

### 5.4 Backend methods

The UI/server must depend on a small backend contract such as:

```python
class Backend(Protocol):
    def load(self, status_callback=None) -> None: ...
    def warmup(self, status_callback=None) -> None: ...
    def generate(self, job, output_dir, progress_callback=None) -> GenerationResult: ...
```

The real and mock backends implement the same contract.

Torch and Diffusers imports must be lazy and restricted to the real backend path so mock tests can run without them installed.

---

## 6. Model loading

### 6.1 Required loader

Use Diffusers `from_single_file()` with the family-selected pipeline:

```text
sd15 -> StableDiffusionPipeline
sdxl -> StableDiffusionXLPipeline
```

Diffusers officially supports single-file `.safetensors` loading for both families.

Reference: https://huggingface.co/docs/diffusers/api/loaders/single_file

### 6.2 Precision/device

Normal real-backend path:

```text
dtype = torch.float16
device = cuda
```

The L4 has 24 GB of GPU memory and Ada-generation Tensor Cores; FP16 is the V1 speed/memory target.

Reference: https://www.nvidia.com/en-us/data-center/l4/

Do not use CPU offload in the normal path. Offload trades latency for memory and conflicts with the primary repeated-generation objective.

### 6.3 SD1.5 safety checker

Disable it:

```text
safety_checker = None
requires_safety_checker = False
```

V1 is a personal model-test harness, not a public generation service.

### 6.4 SDXL watermark

Disable it explicitly:

```text
add_watermarker = False
```

Diffusers otherwise uses `invisible_watermark` by default if that package is present.

Reference: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl

### 6.5 Shared txt2img/img2img components

Do not load a second copy of model weights for img2img.

Create the img2img pipeline from the already-loaded text-to-image components (`from_pipe()` or shared components, depending on the tested Diffusers version). Diffusers documents this as a way to create another pipeline without reallocating model weights.

Reference: https://huggingface.co/docs/diffusers/en/api/pipelines/overview

The two pipeline wrappers must therefore share UNet, VAE, text encoder(s), tokenizer(s), and scheduler configuration state rather than duplicate VRAM-resident weights.

---

## 7. Model-family defaults

These are **V1 starting points**, not claims that every community fine-tune has the same ideal values. Custom checkpoints can prefer different settings; exposed controls remain user-editable.

| Setting | SD 1.5 | SDXL | Reason |
|---|---:|---:|---|
| width | 512 | 1024 | native/base-family scale |
| height | 512 | 1024 | native/base-family scale |
| steps | 25 | 25 | speed/quality starting point for modern samplers |
| CFG | 7.5 | 5.0 | aligns with current Diffusers family defaults |
| sampler | DPM++ 2M SDE Karras | DPM++ 2M SDE Karras | current Diffusers docs call it a strong all-purpose option |
| seed | -1 | -1 | requested behavior |
| images | 1 | 1 | iterative workflow |
| img2img strength | 0.60 | 0.60 | neutral application default; exposed |
| clip skip | `None` | `None` | do not force folklore/model-specific settings globally |

References:

- Stable Diffusion guidance default: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
- SDXL guidance default / API: https://huggingface.co/docs/diffusers/main/api/pipelines/stable_diffusion/stable_diffusion_xl
- Scheduler recommendations: https://huggingface.co/docs/diffusers/main/using-diffusers/schedulers

### 7.1 Hidden settings policy

Keep these unexposed in V1:

- `clip_skip`: `None` for both families.
- SDXL second prompt: reuse main prompt.
- SDXL negative second prompt: reuse main negative prompt.
- SDXL `original_size` / `target_size`: let the pipeline derive normal values from requested dimensions unless a future checkpoint requires a profile.
- SDXL refiner: not used.
- safety checker: disabled.
- watermark: disabled.
- CPU/model offload: disabled.
- VAE slicing/tiling: disabled unless future profiling shows it is required for a specific high-resolution case.

Do not encode "SDXL requires clip skip 2" as a family rule. Current Diffusers exposes clip skip as optional; the official SDXL pipeline does not require a global value of 2.

---

## 8. Sampler contract

Public sampler IDs must be application-owned strings, not Diffusers class names:

```text
dpmpp_2m_karras
dpmpp_2m_sde_karras
euler
euler_a
heun
dpm2_karras
```

UI labels:

```text
DPM++ 2M Karras
DPM++ 2M SDE Karras
Euler
Euler a
Heun
DPM2 Karras
```

Map them internally from the loaded checkpoint scheduler config:

```text
dpmpp_2m_karras
 -> DPMSolverMultistepScheduler
 -> algorithm_type="dpmsolver++"
 -> use_karras_sigmas=True

dpmpp_2m_sde_karras
 -> DPMSolverMultistepScheduler
 -> algorithm_type="sde-dpmsolver++"
 -> use_karras_sigmas=True

euler
 -> EulerDiscreteScheduler

euler_a
 -> EulerAncestralDiscreteScheduler

heun
 -> HeunDiscreteScheduler

dpm2_karras
 -> KDPM2DiscreteScheduler
 -> use_karras_sigmas=True
```

Reference mapping: https://huggingface.co/docs/diffusers/api/schedulers/overview

Scheduler changes are per job and must not reload model weights.

---

## 9. Prompt processing

Prompt processing occurs before the GPU denoising call.

Order:

```text
raw positive/negative template
    -> validate dynamic-brace syntax
    -> choose dynamic alternatives per output image
    -> produce resolved positive/negative strings
    -> convert explicit weighted spans into prompt embeddings
    -> batch embeddings / run pipeline
```

### 9.1 Dynamic prompts

Supported syntax:

```text
a man with {white | black} hair
```

For three requested outputs, each image resolves the choice independently.

Supported combination:

```text
a man with {(white:1.2) | black} hair
```

Possible resolved string:

```text
a man with (white:1.2) hair
```

Rules:

- one brace level only;
- nested braces are a validation error in V1;
- every group must contain at least two alternatives separated by `|`;
- trim surrounding whitespace on alternatives;
- empty alternatives are a validation error;
- multiple separate brace groups in one prompt are allowed;
- dynamic groups are supported in both positive and negative prompts for consistency.

### 9.2 Dynamic choice reproducibility

Prompt selection must be reproducible when the concrete image seed is reproducible.

Use a separate deterministic PRNG derived from each image's concrete seed, for example:

```text
prompt_rng_seed = image_seed XOR fixed_application_salt
```

Do not consume the PyTorch diffusion generator to make brace choices.

Therefore:

```text
seed=123, count=4
```

will reproduce both:

- seed sequence `123,124,125,126`;
- each image's selected dynamic alternatives.

### 9.3 Prompt weighting

Required explicit syntax:

```text
(white hair:1.2)
(background:0.7)
```

Support in positive and negative prompts.

V1 deliberately does **not** promise the entire Automatic1111 parser grammar. In particular, repeated parentheses are not an application-level requirement even if the underlying selected embedding helper happens to understand them.

The user-visible contract is only explicit numeric `(text:weight)`.

Current Diffusers documentation recommends generating weighted embeddings and passing them through `prompt_embeds` / `negative_prompt_embeds`, and currently demonstrates `sd_embed` for SD1.5/SDXL-style weighting. Isolate the chosen weighting library behind `src/prompting.py` so future compatibility changes do not touch server/UI code.

Reference: https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts

### 9.4 Batched weighted prompts

The prompt adapter must support a micro-batch containing different resolved prompts.

Implementation may:

1. deduplicate identical `(positive, negative)` prompt pairs;
2. compute weighted embeddings once per unique pair;
3. repeat/reindex embeddings for identical prompts;
4. pad/align embedding sequence lengths as required by the chosen helper;
5. produce SDXL pooled positive/negative embeddings when required.

This is important because a 10-image job with no dynamic alternatives should not run the text encoders ten unnecessarily identical times.

---

## 10. Seed behavior

Incoming `seed` accepts:

```text
-1                  random per image
0 .. supported max  explicit
```

### 10.1 `seed = -1`

Resolve **all concrete seeds when the job is enqueued**, not when the worker eventually begins.

For count 4, return/log something like:

```text
[481923, 92184, 1827364, 771231]
```

Each image gets its own `torch.Generator`.

### 10.2 Explicit seed

If:

```text
seed = 123
num_images = 4
```

resolve:

```text
123
124
125
126
```

Use one separate `torch.Generator(device="cuda")` per output image, seeded with its concrete seed. Diffusers documents lists of generators for deterministic batched generation.

Reference: https://huggingface.co/docs/diffusers/en/using-diffusers/batched_inference

---

## 11. Input-image preprocessing

User does not receive multiple fit modes in V1.

Required behavior:

1. load image with Pillow;
2. correct EXIF orientation;
3. convert to RGB;
4. preserve aspect ratio;
5. resize until requested width/height is completely covered;
6. center-crop overflow to exactly requested dimensions;
7. never stretch/distort the image.

Example:

```text
source: 1200 x 800
requested: 1024 x 1024

resize preserving ratio -> 1536 x 1024
center crop             -> 1024 x 1024
```

Use high-quality Pillow resampling (LANCZOS) for the resize.

### 11.1 Dimension validation

Backend validation, not just HTML controls:

- width and height must be positive;
- require multiples of 8 at minimum;
- UI increments should use 64px steps for simple/common Stable Diffusion sizing;
- V1 UI range: 256–2048, with a warning/log that large SDXL canvases may OOM.

OOM handling remains authoritative.

---

## 12. Output files

### 12.1 Format

PNG only.

### 12.2 Metadata

Do not write prompt, negative prompt, seed, sampler, model path, or parameter dumps into PNG text chunks.

No JSON sidecars.

Normal PNG structural metadata produced by Pillow is acceptable, but generation data should not be deliberately embedded.

### 12.3 Filename

Use a collision-resistant, readable filename without prompt text, for example:

```text
20261003_162455_job0007_img01_seed123.png
20261003_162455_job0007_img02_seed124.png
```

Filename may contain the seed because that is useful operational identity and is not embedded prompt content.

### 12.4 Live/session information

The server may show the concrete seed and resolved prompt in the in-memory/log UI for the current session. This information is ephemeral unless the user saves terminal logs separately.

---

## 13. Performance strategy on L4

### 13.1 Baseline requirements

The real backend must first work with a conservative fast baseline:

- model loaded once;
- CUDA;
- FP16;
- PyTorch/Diffusers SDPA (automatically used on modern PyTorch where supported);
- no CPU offload;
- no xFormers dependency;
- no third-party FlashAttention wheel;
- no quantization in V1.

Modern Diffusers documents SDPA as built into PyTorch 2 and enabled without xFormers when available.

Reference: https://huggingface.co/docs/diffusers/main/optimization/torch2.0

### 13.2 CUDA allocator

Before Torch import in the real server process:

```text
PYTORCH_ALLOC_CONF=expandable_segments:True
```

Changing batch sizes/resolutions is exactly the kind of allocation pattern PyTorch describes expandable segments as helping.

Reference: https://docs.pytorch.org/docs/stable/notes/cuda

### 13.3 Optional acceleration

Implementation must support trying these without making them required:

- `channels_last` on UNet;
- optionally `channels_last` on VAE;
- `torch.compile` on UNet;
- optionally compiled VAE decode if profiling proves worthwhile.

Diffusers documents compiling UNet as the primary target and documents further speed work by compiling UNet/VAE with channels-last memory format.

References:

- https://huggingface.co/docs/diffusers/main/optimization/torch2.0
- https://huggingface.co/docs/diffusers/optimization/fp16

### 13.4 Compile selection is an implementation benchmark, not an assumption

Phase 8 must test on the actual Colab L4.

At minimum compare:

```text
Profile A
  FP16 + SDPA

Profile B
  A + channels_last UNet + torch.compile(mode="reduce-overhead", fullgraph=True)

Profile C
  A + channels_last UNet/VAE + max-autotune compile
  (only if compile time and stability are acceptable)
```

Measure:

- checkpoint load time;
- compile/warm-up time;
- warm single-image latency;
- second/third single-image latency;
- 10-image total time and images/sec;
- peak allocated/reserved VRAM;
- first generation at a non-default resolution;
- first image-to-image generation.

Pick the default profile that maximizes the repeated personal-testing workflow, not a benchmark-only throughput score.

### 13.5 Warm-up behavior

On server startup:

```text
Flask/UI starts
    -> background backend thread loads model
    -> applies optional optimizations
    -> warm-up generation at model-family default size, batch 1
    -> marks Ready
```

Warm-up does not create a user output file.

The UI is never greyed out. Prompt entry, negative prompt, controls, uploads, input selection, previous-output browsing, and queue submission remain usable.

If a job is submitted before readiness:

```text
queued -> Waiting for model… -> runs automatically when Ready
```

### 13.6 Compile failure

If compile fails but baseline inference is usable:

```text
log warning
fall back to uncompiled pipeline
complete warm-up on baseline
mark Ready
```

Do not convert an optional acceleration failure into a fatal backend error.

### 13.7 Variable resolution and compilation

Current Diffusers guidance notes that a new image shape can trigger a new `torch.compile` graph. V1 accepts this behavior.

Do not depend on dynamic-shape compilation, nightly PyTorch, or manually patched model code in V1.

If a new resolution causes compile latency, surface a log line such as:

```text
Optimizing new tensor shape 768x1024; first generation at this size may be slower.
```

---

## 14. Batch and micro-batch behavior

### 14.1 User contract

`num_images` range:

```text
1..10
```

A requested 10-image job is still one queue item and one logical result set.

### 14.2 True batching

Use Diffusers batch inference, not a Python loop of 10 full independent pipeline calls, whenever VRAM allows.

Diffusers explicitly supports batching and separate generators per output.

Reference: https://huggingface.co/docs/diffusers/en/using-diffusers/batched_inference

### 14.3 OOM fallback

Internal algorithm:

```text
requested = remaining images
candidate = cached working microbatch for (mode,width,height) if present
            else requested

try candidate
    success -> save images; cache candidate as known-good upper size for this key
    CUDA OOM -> clear failed tensors/cache; candidate = max(1, floor(candidate/2)); retry

repeat until all requested outputs are complete
```

Example:

```text
10 requested
 -> batch 10 OOM
 -> batch 5 succeeds
 -> generate 5 + 5
 -> cache 5 for future jobs at this mode/resolution
```

Cache key must distinguish at least:

```text
(txt2img vs img2img, width, height)
```

Model is fixed for process lifetime, so model path need not be in the key.

### 14.4 Partial output safety

Do not expose a half-completed logical job as the latest completed gallery.

Images may be saved as each successful micro-batch finishes, but the server should mark the job completed and switch the main viewer to it only after all requested images succeed.

On a fatal mid-job error:

- leave already-written PNGs in `outputs/` (do not silently delete useful work);
- mark job error;
- log which images completed;
- do not present the partial set as a normal completed latest-job gallery.

### 14.5 Progress

Use Diffusers `callback_on_step_end` to report denoising progress.

Reference: https://huggingface.co/docs/diffusers/using-diffusers/callback

For micro-batched work show both local and total progress, for example:

```text
Batch 1/2 · denoising 14/25
```

Overall progress should advance continuously across micro-batches.

---

## 15. Mock backend

Mock mode is a first-class V1 development tool.

Requirements:

- no Torch import;
- no Diffusers import;
- no network access;
- no model download;
- no GPU requirement;
- same request/result types;
- same queue/server/UI;
- same seed resolution;
- same dynamic prompt resolution;
- same img2img preprocessing code;
- same filesystem paths;
- progress callbacks that advance over simulated steps;
- generate 1–10 deterministic placeholder PNGs with Pillow;
- optionally draw index/seed/model-family text on mock outputs so gallery testing is obvious;
- small configurable sleep to make queue/progress UI testable.

The mock backend should not attempt to emulate Stable Diffusion visually.

---

## 16. Server architecture

### 16.1 Threads

Server process owns:

1. Flask request threads.
2. one background backend-loader thread.
3. one background GPU worker thread.

No second GPU worker.

### 16.2 Backend state

Track an explicit state enum/string:

```text
starting
loading
optimizing
warming
ready
error
```

Also keep a human message:

```text
Loading checkpoint…
Compiling UNet…
Warm-up 1/1…
Ready
```

Use a `threading.Event` (or equivalent) to unblock the worker when the backend reaches `ready` or `error`.

### 16.3 Startup sequence

```text
create Flask app/state/queue
start worker
start backend-loader
start HTTP server immediately
```

The worker may dequeue a job before readiness but must wait for backend completion. Prefer leaving it logically as the active `waiting_for_model` job so the UI clearly distinguishes it from pending queue items.

### 16.4 Load failure

If backend loading fatally fails:

- set backend state `error`;
- expose full concise error in status/log;
- unblock worker;
- mark any waiting active job failed;
- drain/mark pending jobs failed or clear them with an explicit log message;
- future `/api/queue` calls return `503` until process restart.

The browser UI stays available for diagnosis.

### 16.5 Queue

Use a bounded `queue.Queue`.

Recommended V1 pending capacity:

```text
MAX_QUEUE = 5
```

This mirrors the video repo and is enough for personal testing while preventing accidental unbounded work.

`Clear queue` drains pending jobs only. It never interrupts a running/waiting active job.

---

## 17. HTTP API

Exact names may change during implementation, but behavior should remain this small.

### `GET /`

Serve `src/static/index.html`.

### `GET /api/config`

Return immutable/session configuration needed to initialize UI:

```json
{
  "model_family": "sdxl",
  "model_name": "checkpoint.safetensors",
  "defaults": {
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "guidance_scale": 5.0,
    "sampler": "dpmpp_2m_sde_karras",
    "seed": -1,
    "num_images": 1,
    "strength": 0.6
  },
  "samplers": [
    {"id":"dpmpp_2m_karras","label":"DPM++ 2M Karras"},
    {"id":"dpmpp_2m_sde_karras","label":"DPM++ 2M SDE Karras"},
    {"id":"euler","label":"Euler"},
    {"id":"euler_a","label":"Euler a"},
    {"id":"heun","label":"Heun"},
    {"id":"dpm2_karras","label":"DPM2 Karras"}
  ]
}
```

### `GET /api/status`

Return:

```json
{
  "backend_state": "warming",
  "backend_message": "Warm-up 1/1",
  "backend_error": null,
  "queue_length": 2,
  "current_job": {
    "id": 7,
    "status": "waiting_for_model",
    "message": "Waiting for model…",
    "progress": 0.0
  },
  "latest_completed_job": {
    "id": 6,
    "outputs": ["...png", "...png"]
  },
  "outputs_mtime": 1234567890.0,
  "inputs_mtime": 1234567890.0
}
```

Use mtimes/change tokens so the browser does not re-list directories on every 1-second poll.

### `GET /api/inputs`

Newest-first filenames in `inputs/`.

### `POST /api/upload`

Multipart image upload.

- secure filename;
- timestamp/collision-safe destination;
- validate Pillow can open it;
- EXIF transpose/RGB normalization may happen at generation time rather than rewriting upload;
- return stored filename.

### `GET /inputs/<name>`

Serve input image.

### `GET /api/outputs`

Newest-first PNG filenames. May additionally include modification time if convenient for thumbnails.

### `GET /outputs/<name>`

Serve PNG.

### `POST /api/reuse-output`

Input:

```json
{"filename":"...png"}
```

Behavior:

- verify filename exists inside `outputs/`;
- copy bytes into `inputs/` using collision-safe filename;
- do not remove the output;
- return new input filename;
- browser refreshes inputs and auto-selects it.

Do not round-trip through browser canvas/JPEG. Reuse the original PNG bytes.

### `POST /api/queue`

Accept generation request.

Allow while backend states are:

```text
starting/loading/optimizing/warming/ready
```

Reject on fatal `error`.

Resolve seeds and dynamic prompts during enqueue and return concrete seeds.

Validate queue capacity.

### `DELETE /api/queue`

Drain pending jobs; active job remains.

### `DELETE /api/clear-all`

Reject with `409` if:

- active job is running/waiting, **or**
- queue is non-empty.

Then overwrite-before-unlink every normal file in `inputs/` and `outputs/`.

As in the reference repo, document that this is best effort and cannot defeat SSD wear leveling, snapshots, journaling, or cloud infrastructure copies.

---

## 18. UI specification

### 18.1 Visual/technical style

Closely reuse the `videomodeltests` style:

- dark theme;
- centered responsive card around 520px max-width;
- system font;
- plain HTML/CSS/JS;
- inline `<style>` and `<script>`;
- no npm/node/build step;
- status pill with loading/busy/ready/error states;
- polling every ~1 second;
- visible scrollable generation log;
- responsive/mobile-friendly controls.

### 18.2 Header

```text
Stable Diffusion             ● Warming
SDXL · checkpoint.safetensors
```

Status values should distinguish:

```text
Loading
Optimizing
Warming
Ready
Generating
Error
Offline
```

### 18.3 Input image section

- selector of files in `inputs/`;
- `None (text-to-image)` option;
- Upload button;
- small preview thumbnail of selected input;
- uploaded or reused image becomes selected automatically;
- selection persists between jobs until manually changed;
- on page load default to `None` unless the browser has a valid existing selection in-session.

When no image selected:

```text
Text-to-image
```

When image selected:

```text
image.png selected · image-to-image
```

### 18.4 Prompt sections

Positive Prompt:

- textarea;
- reasonable large max character count;
- short hint below or placeholder demonstrating `{a | b}` and `(word:1.2)` syntax.

Negative Prompt:

- textarea;
- same syntax support.

Do not make a prompt syntax editor or builder in V1.

### 18.5 Knobs

Two-column grid similar to reference UI:

```text
Width            Height
Steps            CFG
Sampler          Seed [randomize icon]
Images           Strength (img2img only)
```

Controls:

- width: number, step 64, model default
- height: number, step 64, model default
- steps: integer >=1, default 25
- CFG: number/decimal, default per family
- sampler: select from six choices
- seed: integer, default `-1`
- randomize button: set a concrete random seed in the form; user can type `-1` again for per-run random behavior
- images: integer 1..10
- strength: 0..1 decimal; hide or disable when input is `None`

### 18.6 Generate/queue

Primary button:

```text
Generate image
```

or grammatically adapt to count if trivial:

```text
Generate 10 images
```

Under button:

```text
2 prompts queued                 Clear queue
```

The button stays enabled while model loads/compiles/warms. Submission logs:

```text
Queued job 8 (waiting for model): seeds 123–132
```

### 18.7 Latest result viewer

Always one large image slot.

For a completed 10-image job:

```text
┌──────────────────────────────┐
│                              │
│      selected large image    │
│                              │
└──────────────────────────────┘

[thumb1][thumb2][thumb3][thumb4] ... horizontally scrollable
```

- first output selected automatically when new job completes;
- clicking a thumbnail selects it as the large image;
- selected thumbnail gets visible border;
- large image uses `object-fit: contain`, never visual distortion;
- clicking large image may open the raw PNG in a new browser tab if this works through Colab proxy; otherwise provide a simple `Download/Open` link/button.

For a one-image job, the exact same UI shows one thumbnail.

Do not build a masonry/full-session gallery.

### 18.8 Progress

Preserve reference UI pattern:

```text
Batch 1/2 · denoising 14/25                 28%
████████░░░░░░░░░░░░
```

Backend load states can reuse the same area even when percent is indeterminate:

```text
Compiling UNet…                              —
```

### 18.9 Generation log

`<details open>` like reference UI.

Keep latest ~50–100 lines client-side.

Useful lines:

```text
Server online; model loading in background.
Uploaded portrait.png
Queued job 4: 1 image, seed 921831
Backend: compiling UNet
Backend: warm-up complete
Job 4: DPM++ 2M SDE Karras · 25 steps · 1024x1024
Job 4: denoising 1/25
Job 4: saved outputs/...seed921831.png
Job 4: done in 4.2s
```

Resolved dynamic prompts may be logged per image, because the user explicitly wants prompt experimentation, but they are not persisted into output metadata.

### 18.10 Previous outputs panel

Lower `<details>` panel replacing the reference frame extractor:

```text
▸ Previous outputs

[latest1][latest2][latest3][latest4][latest5]

[ Select any previous output…              ▼ ]

[ Use selected output as input image ]
```

Requirements:

- five newest files displayed as small thumbnails, newest first;
- clicking one may sync the dropdown selection;
- dropdown contains all output filenames, newest first;
- button calls server-side copy endpoint;
- new input is auto-selected in top input selector;
- log the action.

### 18.11 Danger zone

Reuse reference structure/text closely:

```text
▸ Danger zone
Permanently and securely erases every file in inputs/ and outputs/.
This cannot be undone.

[ Clear all inputs & outputs ]
```

Browser confirmation dialog required.

If active/queued work exists, show server rejection and instruct user to clear queue/wait for active job.

---

## 19. CLI

Module invocation:

```bash
python -m src.cli ...
```

### 19.1 `generate`

Example:

```bash
python -m src.cli generate \
  --model models/model.safetensors \
  --model-family sdxl \
  --prompt "portrait photograph" \
  --negative-prompt "blurry" \
  --width 1024 \
  --height 1024 \
  --steps 25 \
  --cfg 5 \
  --sampler dpmpp_2m_sde_karras \
  --seed -1 \
  --images 1 \
  --output-dir outputs
```

Img2img adds:

```bash
--image inputs/source.png --strength 0.6
```

`--mock` should exercise the same command without GPU dependencies. In mock mode, `--model` may be optional or accept a fake path depending on what makes tests simplest.

### 19.2 `serve`

```bash
python -m src.cli serve \
  --model models/model.safetensors \
  --model-family sdxl \
  --port 8000
```

Mock:

```bash
python -m src.cli serve --mock --model-family sdxl --port 8000
```

Real server requires `--model-family` and `--model`.

### 19.3 `doctor`

```bash
python -m src.cli doctor
python -m src.cli doctor --model models/model.safetensors --model-family sdxl
```

See `COLAB_COMPATIBILITY.md`.

### 19.4 No family auto-detection

Do not add:

```text
--model-family auto
```

V1 trusts explicit `sd15` or `sdxl`.

---

## 20. Dependency strategy

See `COLAB_COMPATIBILITY.md` for full policy.

Summary:

- Colab owns Python/CUDA/PyTorch.
- `requirements.txt` supports mock/server/UI development without Torch.
- `requirements-inference.txt` contains Diffusers ecosystem application dependencies but not Torch for normal Colab setup.
- avoid xFormers/custom CUDA extensions in V1;
- use compatible version bounds selected after L4 verification;
- record a known-good Colab snapshot;
- make acceleration capability-detected and fall back gracefully.

As of the current Colab published runtime history, Colab has moved among PyTorch 2.9, 2.10, and 2.11 during 2026, so freezing the entire platform stack in this repository would be brittle.

Reference: https://research.google.com/colaboratory/runtime-version-faq.html

---

## 21. Colab notebook specification

**This notebook is implemented last.**

Do not write the notebook until backend, CLI, server, UI, mock tests, and real L4 tuning are complete. Otherwise setup cells will repeatedly encode assumptions that are still changing.

### 21.1 Notebook role

The notebook contains operational/setup utilities that should **not** live in `src/`:

- runtime preflight;
- dependency installation;
- model download;
- token handling;
- folder creation;
- launching/monitoring server process;
- Colab proxy exposure.

The notebook must not contain a second implementation of image generation.

### 21.2 User inputs

At top:

```python
MODEL_FAMILY = "sdxl"  # @param ["sd15", "sdxl"]
MODEL_URL = ""         # @param {type:"string"}
HF_TOKEN = ""          # optional
CIVITAI_TOKEN = ""     # optional
PORT = 8000
```

The user chooses family explicitly and provides the checkpoint source.

### 21.3 Runtime preflight

Print:

- Python;
- Torch;
- CUDA available;
- GPU model;
- GPU VRAM.

Warn, but do not necessarily fail, if GPU is not L4.

Fail real mode if CUDA is unavailable.

### 21.4 Install

- install repo local/mock requirements;
- install inference application requirements;
- avoid reinstalling Torch by default;
- rerun/verify `doctor` after installation.

### 21.5 Download helpers

The notebook may define functions such as:

```python
download_huggingface_checkpoint(...)
download_civitai_checkpoint(...)
```

These are notebook utilities, not `src/` application code.

#### Hugging Face

Prefer `huggingface_hub.hf_hub_download` for repo ID + filename because it handles cache/versioning and tokens cleanly.

Reference: https://huggingface.co/docs/huggingface_hub/main/en/guides/download

If V1 accepts a direct HF file URL, normalize `/blob/` to a downloadable form or use Hub parsing rather than depending on browser HTML.

#### Civitai

Support a Civitai model-version `downloadUrl` / `https://civitai.com/api/download/models/{version_id}` with optional Bearer token. Honor `Content-Disposition` so the actual filename is retained.

Reference: https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md

The helper should stream to disk and display download progress where practical; large checkpoint downloads must not be read entirely into RAM.

### 21.6 Checkpoint destination

Exactly one active model file expected:

```text
models/<downloaded-name>.safetensors
```

Validate extension. Pass exact path to CLI.

Do not add automatic model directory discovery to V1; notebook already knows the downloaded path.

### 21.7 Launch

Start:

```bash
python -m src.cli serve \
  --model "$MODEL_PATH" \
  --model-family "$MODEL_FAMILY" \
  --port 8000
```

as a background process while preserving logs.

Because the HTTP server itself starts before GPU warm-up, the next cell does not need to wait for backend Ready; it only needs to wait until the port responds.

### 21.8 Proxy

V1 uses only Colab's built-in kernel proxy.

Prefer the currently supported `serve_kernel_port_as_iframe()` helper rather than the deprecated `serve_kernel_port_as_window()`.

Reference: https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py

No Cloudflare/ngrok cell in V1.

---

## 22. Validation and tests

### 22.1 Entire normal test suite must run without GPU packages

A fresh small development machine should be able to run:

```bash
pytest
```

without installing Torch/Diffusers.

Any real-backend tests must be separated/marked and are not part of normal CI/local success criteria.

### 22.2 Prompt tests

Required examples:

```text
"a {white | black} cat"
```

- count 10 produces only valid alternatives;
- fixed seed reproduces choices;
- different image seeds can produce different choices.

```text
"a {(white:1.2) | black} cat"
```

- brace resolution preserves weight syntax.

Reject:

```text
"a {red | {blue | green}} cat"
"a {red | } cat"
"a {} cat"
```

### 22.3 Seed tests

- `-1`, 10 images => 10 concrete valid seeds (collisions astronomically unlikely; implementation may explicitly ensure uniqueness within one job).
- `123`, 4 => `[123,124,125,126]`.
- same explicit job => same dynamic prompt choices.

### 22.4 Preprocess tests

Input 1200×800 -> 1024×1024 must crop, not distort.

Test portrait and landscape sources.

Test EXIF orientation.

### 22.5 Mock backend tests

- 1 image;
- 10 images;
- txt2img;
- img2img;
- progress callback;
- output naming;
- output PNG valid;
- result order matches seed order.

### 22.6 Server tests

- UI responds before mock backend warm-up finishes;
- enqueue while loading accepted;
- queued job begins when ready;
- queue capacity enforced;
- clear queue does not cancel active job;
- upload validation;
- reuse-output copies file to inputs;
- latest completed job returns ordered output list;
- output/input listing newest-first;
- clear-all rejected with active job;
- clear-all rejected with pending queue;
- clear-all works when idle;
- fatal backend load error visible through status and rejects new queue requests.

### 22.7 UI manual test checklist in mock mode

- mobile-width layout;
- prompt and negative prompt editing;
- seed randomize;
- img2img strength appears/disappears with input selection;
- Generate usable during loading;
- status transitions;
- queue count;
- progress/log updates;
- 1-image viewer;
- 10-image thumbnail band scroll/select;
- latest five previous-output thumbnails;
- arbitrary old output from dropdown;
- reuse output as input;
- clear queue;
- danger-zone confirmation/error states.

---

## 23. Real L4 verification (Phase 8)

Before notebook finalization, test at least one known-good SD1.5 `.safetensors` and one known-good SDXL `.safetensors` on an L4.

### 23.1 Functional matrix

For each family:

- txt2img count 1;
- txt2img count 10;
- img2img count 1;
- img2img count 10 or maximum practical count;
- all six samplers;
- seed `-1`;
- fixed seed sequence;
- weighted prompt;
- dynamic prompt;
- combined dynamic + weighting;
- non-square resolution;
- output reuse into img2img;
- repeated single-image requests without reload.

### 23.2 Performance matrix

At family default resolution and 25 steps:

Measure at least:

```text
cold checkpoint load
compile/warm-up
warm generation #1
warm generation #2
warm generation #3
batch 5
batch 10
peak GPU allocated/reserved
```

Run against optimization profiles A/B/C described above.

### 23.3 Acceptance principle

The selected default optimization profile should have:

- stable repeated generations;
- no model reload between requests;
- meaningful warm-latency improvement over baseline if compilation is enabled;
- tolerable compile startup cost for a personal Colab session;
- graceful OOM batch fallback;
- no external CUDA-extension dependency.

Do not set an arbitrary hard seconds-per-image target before testing the actual L4/checkpoints. Record the measured numbers in `compat/known-good-colab.md`.

---

## 24. Logging

### 24.1 Terminal/server logging

Include concise structured information:

```text
backend state transitions
model path basename + family
model load duration
optimization success/fallback
warm-up duration
queue enqueue/drain
job id
mode txt2img/img2img
requested image count
resolved seed list
sampler/steps/size/CFG/strength
micro-batch attempts and OOM fallback
per-job elapsed time
output filenames
errors with traceback in terminal
```

Do not log authentication tokens or model download bearer headers.

### 24.2 Browser log

Shorter user-facing subset, capped in memory like the reference UI.

---

## 25. Security/file safety boundaries

This is a personal Colab tool, so V1 does not add app authentication.

Still enforce:

- `secure_filename` on uploads;
- no user-supplied arbitrary filesystem paths through HTTP API;
- all input/output names resolved only under known directories;
- extension/type validation for uploads;
- model path provided only at process startup, not through browser;
- tokens only in notebook process variables/environment and never passed to browser;
- generated UI HTML should use `textContent`, not unsafe `innerHTML`, for prompt/log strings;
- clear-all only touches normal files inside `inputs/` and `outputs/`.

The Colab proxy is not a substitute for a public production authentication layer, but V1 is not intended to be publicly hosted.

---

## 26. Implementation phases

### Phase 1 — core pure-Python behavior

Implement/test:

- request validation;
- model family enum/literal;
- seed resolution;
- one-level dynamic prompt expansion;
- image preprocessing;
- filename generation.

No Torch.

### Phase 2 — mock backend

Implement mock `load`, `warmup`, `generate` and progress.

### Phase 3 — real Diffusers backend

Implement:

- lazy imports;
- explicit family pipeline class;
- single-file load;
- safety checker/watermark disable;
- shared img2img components;
- sampler mapping;
- weighted prompt adapter;
- per-image generators;
- batching;
- OOM fallback;
- progress callbacks;
- output save.

Start with baseline, not compile optimization.

### Phase 4 — CLI

Implement `doctor`, `generate`, `serve`.

### Phase 5 — queue + HTTP API

Implement loader thread, worker, filesystem endpoints and status.

### Phase 6 — UI

Adapt reference UI.

### Phase 7 — local/mock verification

All standard tests pass without GPU dependencies.

### Phase 8 — L4 tuning

Benchmark/select acceleration profile. Update defaults only based on evidence. Write `compat/known-good-colab.md`.

### Phase 9 — Colab notebook LAST

Implement preflight, installs, downloads, server launch, and built-in proxy using the now-stable CLI/API contract.

---

## 27. V1 acceptance criteria

V1 is complete when all are true:

1. `pytest` passes on a non-GPU machine without Torch/Diffusers installed.
2. `serve --mock` provides the complete UI and all user flows.
3. Browser UI loads before backend warm-up is complete and can queue a generation during warm-up.
4. Real SD1.5 single-file checkpoint loads on Colab L4 and can run txt2img + img2img.
5. Real SDXL single-file checkpoint loads on Colab L4 and can run txt2img + img2img.
6. Repeated jobs reuse the same resident model; no checkpoint reload occurs.
7. Count 10 uses batching and can automatically fall back to smaller micro-batches on OOM.
8. Seed behavior matches `-1` random-per-image and fixed sequential seeds.
9. `(text:weight)` works for positive and negative prompts.
10. One-level `{a | b}` works independently and reproducibly per image.
11. All six sampler entries work for both tested families.
12. Input images are fit without distortion.
13. Latest job viewer uses one large selected image + scrollable full-job thumbnail strip.
14. Previous outputs shows five latest thumbnails + full filename selector + reuse-as-input.
15. Output files are PNGs with no deliberate prompt/parameter metadata.
16. Clear queue preserves active generation.
17. Clear-all is blocked when active or queued work exists and performs best-effort overwrite/unlink when idle.
18. Safety checker is disabled for SD1.5.
19. SDXL invisible watermark is disabled.
20. Colab notebook does not normally reinstall Torch, runs doctor, downloads one checkpoint, launches server, and exposes UI using only Colab's built-in proxy.
21. Optional acceleration failure falls back to working baseline inference.
22. A known-good L4 compatibility snapshot and benchmark results are committed after testing.

---

## 28. Future V2 candidates

V1 architecture should permit, but not implement:

- GGUF through a separate backend adapter if desired;
- Diffusers-directory model input;
- model-specific profiles/defaults from Civitai metadata;
- Cloudflare tunnel;
- LoRAs;
- ControlNet;
- inpainting;
- multiple loaded models/model switching;
- richer prompt syntax/nested dynamic groups;
- persistent session history/parameter database;
- metadata/sidecar opt-in;
- Docker/RunPod packaging;
- optional runtime-specific optimized engines;
- stronger full-tab Colab proxy handling if Google stabilizes a supported API.

The V1 public application interfaces (`GenerationRequest`, sampler IDs, CLI flags, HTTP JSON fields) should stay independent of Diffusers class names so a future backend can be replaced without rewriting the UI.

---

## 29. Research references used by this specification

- Diffusers single-file loading: https://huggingface.co/docs/diffusers/api/loaders/single_file
- Diffusers single-file formats: https://huggingface.co/docs/diffusers/using-diffusers/other-formats
- Diffusers PyTorch 2 / SDPA / `torch.compile`: https://huggingface.co/docs/diffusers/main/optimization/torch2.0
- Diffusers acceleration guide: https://huggingface.co/docs/diffusers/optimization/fp16
- Diffusers batch inference: https://huggingface.co/docs/diffusers/en/using-diffusers/batched_inference
- Diffusers scheduler guidance: https://huggingface.co/docs/diffusers/main/using-diffusers/schedulers
- Diffusers A1111/k-diffusion scheduler mapping: https://huggingface.co/docs/diffusers/api/schedulers/overview
- Diffusers prompt weighting: https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts
- Diffusers callbacks: https://huggingface.co/docs/diffusers/using-diffusers/callback
- Stable Diffusion API: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
- SDXL API/watermark behavior: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl
- Pipeline component reuse: https://huggingface.co/docs/diffusers/en/api/pipelines/overview
- NVIDIA L4 specifications: https://www.nvidia.com/en-us/data-center/l4/
- PyTorch CUDA allocator / expandable segments: https://docs.pytorch.org/docs/stable/notes/cuda
- Colab past runtime versions: https://research.google.com/colaboratory/runtime-version-faq.html
- Colab FAQ/resource behavior: https://research.google.com/colaboratory/faq.html
- Colab kernel-port helpers: https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py
- Hugging Face Hub downloads: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
- Civitai model-version API/download URL: https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md
