# Colab Runtime and Dependency Compatibility Strategy

**Purpose:** keep the V1 repository usable as Google Colab updates Python, PyTorch, CUDA-facing libraries, and browser proxy behavior.

This document is part of the implementation contract. It is not a generic deployment guide.

## 1. Core rule: Colab owns the platform stack

Treat these as host/platform dependencies supplied by Colab:

- Linux runtime
- Python
- NVIDIA driver
- CUDA runtime exposed through the installed PyTorch build
- PyTorch
- core scientific packages when the preinstalled version is already compatible

Do **not** make a normal Colab setup cell start by upgrading/replacing Torch, torchvision, or CUDA libraries. The notebook should first use the CUDA-enabled PyTorch build already paired with the current Colab image.

Colab explicitly updates its runtime images frequently and keeps past runtime versions available for a limited period. As of October 2026, its published past-runtime list shows Python 3.12 with PyTorch versions changing from 2.9.0 to 2.10.0 to 2.11.0 across the 2026.01, 2026.04, and 2026.07 images. This is exactly why the repo should avoid pretending it controls the whole platform environment.

Reference: Google Colab, "Past Runtime Versions" — https://research.google.com/colaboratory/runtime-version-faq.html

## 2. Project-owned dependency layers

Use two small dependency manifests rather than one frozen GPU environment.

### `requirements.txt` — local/mock application

Contains only dependencies needed for the CLI/server/UI/mock path, for example:

```text
flask
pillow
requests
```

Development/test dependencies may live in `pyproject.toml` or a dev group:

```text
pytest
ruff
```

There must be no required `torch`, `diffusers`, CUDA extension, or model download for the normal local test suite.

### `requirements-inference.txt` — real inference adapter

Contains app-level inference libraries, for example:

```text
diffusers
transformers
accelerate
safetensors
huggingface_hub
# weighted-prompt adapter dependency selected during implementation
```

**Do not list Torch here for Colab.**

The exact lower/upper version bounds are chosen only after Phase 8 L4 testing. Use compatible ranges, not a fully frozen transitive environment. The exception may be a small pure-Python prompt-weighting dependency if its prompt semantics must be held stable; if pinned, isolate it behind `src/prompting.py` so it can be replaced independently.

## 3. Known-good runtime snapshot

After real L4 testing, add a file such as:

```text
compat/known-good-colab.md
```

It records, but does not automatically force:

- date tested
- Colab runtime version if identifiable
- Python version
- Torch version
- CUDA availability reported by Torch
- GPU model and VRAM
- Diffusers version
- Transformers version
- Accelerate version
- prompt-weighting dependency/version or commit
- whether SDPA worked
- whether `torch.compile` worked
- selected compile mode
- SD1.5 benchmark numbers
- SDXL benchmark numbers

The purpose is recovery and diagnosis. If a new Colab image breaks the project, a user may temporarily select the recent known-good runtime while the compatibility bounds are updated. Colab currently documents that past runtime versions remain selectable for about one year.

## 4. `doctor` is required

Implement:

```bash
python -m src.cli doctor
```

Mock/local mode must run without importing Torch. GPU diagnostics are loaded lazily only when requested/available.

On a GPU environment, print at least:

```text
Python:                 3.x
PyTorch:                x.y.z
CUDA available:         yes/no
Torch CUDA version:     ...
GPU:                    NVIDIA L4
GPU memory:             ... GiB
Diffusers:              ...
Transformers:           ...
Accelerate:             ...
Safetensors:            ...
SDPA:                   available/unavailable
channels_last:          available/unavailable
torch.compile:          available/unavailable
compile smoke check:    pass/fail/not-run
model family:           sd15/sdxl (when supplied)
checkpoint exists:      yes/no (when supplied)
```

A mismatch should produce a specific message rather than a giant import traceback where practical.

Hard fail only on capabilities that are required for the selected command. For example:

- local/mock serve: no CUDA requirement
- real generate/serve: CUDA, Torch, Diffusers, checkpoint required
- `torch.compile`: **not required**; warn and fall back

## 5. Capability detection beats version detection

Version numbers belong in diagnostics; behavior should prefer direct capability checks.

Examples:

- Is `torch.cuda.is_available()` true?
- Can an FP16 CUDA tensor be allocated?
- Does the current Diffusers pipeline load this single-file checkpoint?
- Is PyTorch SDPA available? Modern Diffusers uses PyTorch 2 SDPA by default when supported.
- Does `torch.compile` exist and can the UNet warm-up succeed?
- Does `channels_last` conversion succeed?
- Can the selected scheduler be constructed from the loaded scheduler config?

Do not silently select a different model family if loading fails. `sd15` versus `sdxl` is an explicit trusted user input in V1.

## 6. Optimization fallback hierarchy

The inference backend should conceptually have this ladder:

```text
required baseline
    FP16 CUDA
    standard Diffusers pipeline
    PyTorch SDPA where automatically available

optional acceleration
    channels_last
    torch.compile(UNet)
    optional compiled VAE decode if L4 benchmarking proves worthwhile

fallback
    if an optional optimization fails, log it and continue on the baseline
```

Do not make xFormers, FlashAttention pip packages, bitsandbytes, custom CUDA kernels, or other compiled third-party extensions part of V1. PyTorch 2/Diffusers already provide SDPA without xFormers, and avoiding extra binary wheels substantially reduces Colab compatibility risk.

References:

- Diffusers PyTorch 2 / SDPA / compile: https://huggingface.co/docs/diffusers/main/optimization/torch2.0
- Diffusers acceleration guide: https://huggingface.co/docs/diffusers/optimization/fp16

## 7. CUDA allocator setup

Before Torch is imported in the real server process, set:

```text
PYTORCH_ALLOC_CONF=expandable_segments:True
```

PyTorch documents expandable segments as useful for workloads where allocation sizes change, such as changing batch size. That aligns with this project's automatic micro-batching and variable resolutions.

`PYTORCH_CUDA_ALLOC_CONF` is a backward-compatible alias, but use the current `PYTORCH_ALLOC_CONF` name in new code/notebook cells.

Reference: https://docs.pytorch.org/docs/stable/notes/cuda

## 8. Compile policy

`torch.compile` is valuable for the expected "load once, generate repeatedly" usage, but it must not be treated as universally safe.

Rules:

1. Load the pipeline successfully before attempting compile.
2. Convert the UNet to `channels_last` if supported.
3. Try the compile profile selected by Phase 8 benchmarking.
4. Warm up at the family default shape, batch 1:
   - SD1.5: 512×512
   - SDXL: 1024×1024
5. If compilation or warm-up fails, restore/use the uncompiled model, log the reason, and mark the backend ready.
6. The UI remains usable during load/compile/warm-up and may queue jobs.
7. Different image shapes can trigger recompilation. This is acceptable in V1 and should be visible in the log if it causes a noticeable first-run pause.
8. Do not depend on dynamic-shape compilation in V1. Current Diffusers guidance notes that dynamic compilation can help avoid recompiles but is not uniformly beneficial and has stronger support on nightly PyTorch; this conflicts with the future-proofing goal of using Colab's stable platform build.

Phase 8 should compare at least:

```text
A. FP16 + SDPA baseline
B. A + channels_last + UNet compile, reduce-overhead
C. A + channels_last + UNet/VAE compile, max-autotune (only if compile cost is acceptable)
```

Pick the fastest **repeated-generation** profile that is stable on the L4 without making startup unreasonable. Single-image warm latency is the primary metric; 10-image batch throughput is secondary.

## 9. Do not persist compiled artifacts across Colab sessions in V1

A persistent TorchInductor cache in Drive is tempting, but compiled artifacts can depend on PyTorch, CUDA/toolchain details, graph shapes, and hardware. Since Colab updates regularly, cross-session cache invalidation adds more failure modes than it removes.

Keep compiler caches ephemeral in `/content` for V1.

## 10. Single-file loading strategy

V1 accepts a local `.safetensors` checkpoint and an explicit family:

```text
sd15
sdxl
```

Use the family to select the pipeline class rather than trying to infer the family from checkpoint keys.

Diffusers supports `from_single_file()` for Stable Diffusion and SDXL pipelines. It may use model configuration from the Hub when loading a local checkpoint. Because the notebook already requires network access to download a model, V1 may allow configuration fetch/cache during initial model load.

References:

- https://huggingface.co/docs/diffusers/api/loaders/single_file
- https://huggingface.co/docs/diffusers/using-diffusers/other-formats

If future testing finds that a specific checkpoint needs a custom config, that is a V2 compatibility profile rather than a reason to reintroduce family auto-detection.

## 11. Notebook preflight and install order

The final notebook should be linear and readable.

### Cell group A — user configuration

Colab form parameters or plainly editable variables:

```text
MODEL_FAMILY = "sdxl"   # exactly "sd15" or "sdxl"
MODEL_URL = "..."
HF_TOKEN = ""           # optional
CIVITAI_TOKEN = ""      # optional
PORT = 8000
```

Never commit tokens into the repo or notebook defaults.

### Cell group B — runtime preflight

Before installs:

- print Python version
- import the preinstalled Torch if available
- print Torch version
- print CUDA availability
- print GPU name/VRAM
- warn if GPU is not L4, but do not hard-fail solely because another CUDA GPU was assigned

### Cell group C — project installation

Install local/mock requirements and inference application libraries.

Do **not** run a blanket:

```text
pip install --upgrade torch torchvision
```

unless a future compatibility note explicitly says the current Colab platform is broken and a tested replacement is required.

### Cell group D — doctor

Run `python -m src.cli doctor` and stop the notebook flow on a required-capability failure.

### Cell group E — checkpoint download

Create `models/`, `inputs/`, and `outputs/`.

Support:

- Hugging Face: preferably `hf_hub_download` when repo ID + filename can be derived/provided; otherwise a direct resolved file URL with optional Bearer token.
- Civitai: direct version download endpoint / `downloadUrl`, with optional Bearer token and `Content-Disposition` filename handling.

Verify:

- download completed
- file extension is `.safetensors`
- file size is nonzero/reasonable
- only one active checkpoint path is passed to `serve`

References:

- Hugging Face downloads: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
- Civitai model-version API: https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md

### Cell group F — launch server

Launch the server as a background process with stdout/stderr visible or tailed into the notebook. Do not hide backend errors.

### Cell group G — Colab proxy

Use only Colab's built-in kernel proxy in V1. No Cloudflare/ngrok.

The current `google.colab.output` implementation marks `serve_kernel_port_as_window()` as deprecated because browser security changes can break it, and recommends `serve_kernel_port_as_iframe()` instead. Therefore V1 should prefer the supported iframe helper. If implementation testing confirms a reliable full-tab proxy link on the current runtime, it may additionally render one, but the iframe path is the compatibility baseline.

Reference: https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py

## 12. Colab operational caveat

Colab resources and available GPU types can change. Google also notes that managed free-tier runtimes may terminate sessions that primarily bypass the notebook UI to interact through a web UI; paid compute changes some of those restrictions. This repository should not attempt to bypass Colab controls or idle/resource policies.

Reference: https://research.google.com/colaboratory/faq.html

## 13. Updating compatibility later

When Colab changes:

1. Run `doctor` on the new runtime.
2. Run the mock suite (should remain independent of GPU stack).
3. Load one known SD1.5 checkpoint.
4. Load one known SDXL checkpoint.
5. Run one txt2img and one img2img generation for each family.
6. Run batch counts 1 and 10 at default resolution.
7. Verify all six sampler mappings construct and generate.
8. Verify prompt weighting and dynamic prompt expansion.
9. Verify compile fallback and the selected optimization profile.
10. Update dependency bounds only as narrowly as necessary.
11. Update `compat/known-good-colab.md` after passing.

This keeps maintenance focused on the small version-sensitive inference adapter instead of repeatedly rebuilding the repository around a new Colab image.
