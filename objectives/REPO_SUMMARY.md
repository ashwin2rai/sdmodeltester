# Stable Diffusion Image Test UI — Repository Summary

**Status:** V1 implementation specification summary  
**Target runtime:** Google Colab with an NVIDIA L4 GPU  
**Model scope:** one user-selected SD 1.5 or SDXL `.safetensors` checkpoint per session  
**Primary goal:** fast, iterative image-model testing through a minimal browser UI

## What this repository is

This repository is a deliberately small image-generation test harness for **Stable Diffusion 1.5 and SDXL single-file `.safetensors` checkpoints**. It is inspired by `videomodeltests`, but it is intentionally narrower and simpler.

The repository contains a real inference backend, a CLI, a tiny HTTP server, and a browser UI. However, **the backend is not the product by itself**. Its main purpose is to make a Stable Diffusion checkpoint easy to load once on a Google Colab L4, keep resident in GPU memory, and then drive repeatedly from a lightweight web UI exposed through Colab's built-in kernel proxy.

The expected workflow is:

```text
open Colab notebook
    -> choose SD 1.5 or SDXL
    -> provide one Hugging Face or Civitai checkpoint link
    -> notebook installs only the missing app-level dependencies
    -> checkpoint downloads into models/
    -> server starts immediately
    -> model loads / optimizes / warms in the background
    -> open the built-in Colab-proxied UI
    -> iterate on prompts and img2img generations
```

The repository should optimize for a personal experimentation loop where the most common action is **generate one image, tweak the prompt, generate another image, repeat**. Generating 5–10 variations from one prompt is also supported and should use real GPU batching where memory permits.

## Design principles

1. **One model per session.** The checkpoint is selected at process startup and remains resident. There is no model switcher in the UI and no runtime model unloading/reloading path.
2. **Speed over format breadth.** V1 supports only single `.safetensors` checkpoints. GGUF and Diffusers-directory models are explicitly deferred.
3. **Colab-first, but not Colab-coupled.** Core inference/server code contains no Colab-specific setup. The notebook handles installation, download, runtime checks, launch, and proxy exposure.
4. **Mock-first development.** Almost all backend/API/UI development must be testable on a machine with no CUDA, no Torch, no Diffusers, and no checkpoint. The mock backend implements the same application contract and creates placeholder PNGs.
5. **Keep the model in memory.** Loading happens once. Text-to-image and image-to-image pipelines share the same loaded components. Requests are serialized through one GPU worker queue.
6. **Use batching without exposing batching complexity.** A job may request 1–10 images. The backend batches them, automatically reduces the micro-batch size after CUDA OOM, and remembers the working size for that resolution/mode for the rest of the session.
7. **Optimizations must be optional.** FP16 and PyTorch SDPA form the stable fast baseline. `channels_last`, `torch.compile`, compiled VAE decode, and similar improvements are enabled only when supported and verified. Failure of an optimization must fall back to the baseline rather than break the app.
8. **Preserve the minimal `videomodeltests` interaction style.** One dark responsive card, plain HTML/CSS/JS, no frontend build system, ~1 second polling, visible queue/progress/logging, and a danger-zone clear-all action.

## Generation features

V1 supports both:

- **text → image**
- **image + text → image**

The UI exposes prompt, negative prompt, width, height, steps, CFG/guidance, sampler, seed, image count, and img2img strength. The initial values are model-family aware: SD 1.5 starts at 512×512 and SDXL at 1024×1024. The user may override exposed values.

Prompt conveniences are part of V1:

```text
(white hair:1.2)               # explicit prompt weight
{white | black | silver}       # choose one alternative per output image
{(white:1.2) | black} hair     # dynamic choice and weighting together
```

Brace expansion is one level only. For a multi-image job it is resolved independently per image. Seed `-1` means a new random concrete seed for every output. A fixed seed creates a deterministic sequence: requesting four images from seed `123` uses `123, 124, 125, 126`.

## UI result flow

The latest completed job has one large image viewer. Under it is a horizontally scrollable strip containing every image from that job. Selecting a thumbnail changes the large image.

A lower **Previous outputs** panel reuses the useful idea from the video repository's frame-extraction panel. It shows thumbnails for the five newest output files and a dropdown containing every output PNG. Any previous output can be copied into `inputs/` and automatically selected as the source for the next img2img generation.

The UI also retains an exposed generation log, queue count, clear-pending-queue action, progress bar, upload/input selector, and secure/best-effort **Clear all inputs & outputs** action. Clear-all is blocked while a job is active or the pending queue is non-empty.

## Colab and dependency philosophy

Colab's Python, CUDA, NVIDIA driver, and preinstalled PyTorch are treated as the **host platform**, not as dependencies this project owns. V1 should not blindly reinstall or upgrade Torch in the notebook.

The repository owns only its small application-level dependency set. GPU optimizations are capability-detected. A `doctor` CLI command reports Python, Torch, CUDA, GPU, Diffusers/Transformers versions, and which acceleration features are usable.

A tested runtime snapshot is recorded after real L4 verification, but it is a recovery reference rather than the default installer. This protects the project from regular Colab runtime updates without freezing the entire environment forever.

## Development order

The Colab notebook is intentionally built **last**:

```text
1. request types, seeds, dynamic prompts, preprocessing
2. mock backend
3. real Diffusers backend
4. CLI
5. queue and HTTP API
6. browser UI
7. local mock/CPU tests
8. real L4 profiling and optimization
9. Colab notebook + runtime compatibility snapshot
```

The full behavior, interfaces, acceptance criteria, performance strategy, and out-of-scope items are defined in `SPEC.md`. Colab dependency/runtime policy is expanded in `COLAB_COMPATIBILITY.md`.
