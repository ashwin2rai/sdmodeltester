"""Request/result types, pure-Python job resolution, and generation backends.

This module must stay importable without torch/diffusers. The real Diffusers
backend (Phase 3) imports them lazily inside its methods.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, get_args

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.prompting import PromptSyntaxError, resolve_prompt_pair, strip_weights, validate_template

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

ModelFamily = Literal["sd15", "sdxl", "zimage"]
MODEL_FAMILIES: tuple[str, ...] = get_args(ModelFamily)

Mode = Literal["txt2img", "img2img"]

# Public sampler IDs are application-owned (SPEC §5); never Diffusers class names.
SAMPLERS: tuple[tuple[str, str], ...] = (
    ("dpmpp_2m_karras", "DPM++ 2M Karras"),
    ("dpmpp_2m_sde_karras", "DPM++ 2M SDE Karras"),
    ("euler", "Euler"),
    ("euler_a", "Euler a"),
    ("heun", "Heun"),
    ("dpm2_karras", "DPM2 Karras"),
    ("flow_euler", "Flow Euler"),
)
SAMPLER_IDS: tuple[str, ...] = tuple(sid for sid, _ in SAMPLERS)
SAMPLER_LABELS: dict[str, str] = dict(SAMPLERS)

# Sampler ID -> (diffusers scheduler class name, from_config overrides). SPEC §5.
# Overrides are explicit (incl. use_karras_sigmas=False) so settings in the checkpoint's
# own scheduler config can't silently turn e.g. "Euler" into "Euler Karras".
SAMPLER_SCHEDULERS: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True},
    ),
    "dpmpp_2m_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 2, "use_karras_sigmas": True},
    ),
    "euler": ("EulerDiscreteScheduler", {"use_karras_sigmas": False}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {"use_karras_sigmas": False}),
    "dpm2_karras": ("KDPM2DiscreteScheduler", {"use_karras_sigmas": True}),
    # Z-Image is a flow-matching model: keep its own scheduler config (incl. shift).
    "flow_euler": ("FlowMatchEulerDiscreteScheduler", {}),
}
# The SD samplers are noise-prediction schedulers; flow matching needs its own.
_SD_SAMPLERS = tuple(sid for sid in SAMPLER_IDS if sid != "flow_euler")

MAX_IMAGES = 10
MIN_DIMENSION = 256
MAX_DIMENSION = 2048
DIMENSION_MULTIPLE = 8
MAX_STEPS = 150
MAX_GUIDANCE = 30.0
MAX_PROMPT_CHARS = 4000
# Matches the common A1111/Civitai seed range; sequential seeds must stay inside it.
MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class FamilyDefaults:
    width: int
    height: int
    steps: int
    guidance_scale: float
    sampler: str
    seed: int
    num_images: int
    strength: float


@dataclass(frozen=True)
class FamilySpec:
    """Everything that differs per model family, apart from how its pipeline is loaded."""

    defaults: FamilyDefaults  # served as-is by /api/config
    samplers: tuple[str, ...]
    dtype: str = "float16"
    dimension_multiple: int = DIMENSION_MULTIPLE
    # False: the text encoder has no (text:1.2) syntax, so weights are stripped at enqueue
    # and the pipeline encodes the plain prompts itself.
    prompt_weights: bool = True
    denoiser: str = "unet"


FAMILIES: dict[str, FamilySpec] = {
    "sd15": FamilySpec(
        FamilyDefaults(512, 512, 25, 7.5, "dpmpp_2m_sde_karras", -1, 1, 0.6), _SD_SAMPLERS
    ),
    "sdxl": FamilySpec(
        FamilyDefaults(1024, 1024, 25, 5.0, "dpmpp_2m_sde_karras", -1, 1, 0.6), _SD_SAMPLERS
    ),
    # Turbo: 9 steps = 8 transformer passes; distilled, so guidance off (0 = no CFG pass).
    # Sizes: VAE factor 8 x patch 2 (the pipeline silently floors anything else); Qwen3 LLM
    # text encoder; BF16 (FP16 overflows).
    "zimage": FamilySpec(
        FamilyDefaults(1024, 1024, 9, 0.0, "flow_euler", -1, 1, 0.6),
        ("flow_euler",),
        dtype="bfloat16",
        dimension_multiple=16,
        prompt_weights=False,
        denoiser="transformer",
    ),
}


class ValidationError(ValueError):
    """A generation request (or a parameter of it) is invalid."""


def check_family(family: str) -> ModelFamily:
    if family not in MODEL_FAMILIES:
        raise ValidationError(
            f"model family must be one of {', '.join(MODEL_FAMILIES)} (got {family!r})"
        )
    return family  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Request / job / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    sampler: str
    seed: int  # -1 or 0..MAX_SEED on incoming request
    num_images: int  # 1..MAX_IMAGES
    input_image: Path | None = None
    strength: float | None = None  # required only when input_image exists

    @property
    def mode(self) -> Mode:
        return "img2img" if self.input_image is not None else "txt2img"


@dataclass(frozen=True)
class ResolvedImageSpec:
    index: int  # 0-based position within the job
    seed: int
    prompt: str
    negative_prompt: str


@dataclass(frozen=True)
class ResolvedGenerationJob:
    request: GenerationRequest
    images: tuple[ResolvedImageSpec, ...]
    job_id: int = 0
    # Human-readable backend corrections applied to the request (for the log).
    corrections: tuple[str, ...] = ()

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(img.seed for img in self.images)


@dataclass(frozen=True)
class GenerationResult:
    output_paths: tuple[Path, ...]
    seeds: tuple[int, ...]
    resolved_prompts: tuple[str, ...]
    elapsed_seconds: float


class GenerationError(RuntimeError):
    """A job failed; ``completed_paths`` lists PNGs already written (left in place)."""

    def __init__(self, message: str, completed_paths: Sequence[Path] = ()):
        super().__init__(message)
        self.completed_paths = tuple(completed_paths)


# (state, human message), e.g. ("loading", "Loading checkpoint…")
StatusCallback = Callable[[str, str], None]
# (overall fraction 0..1, human message), e.g. (0.28, "Batch 1/2 · denoising 14/25")
ProgressCallback = Callable[[float, str], None]


class Backend(Protocol):
    family: ModelFamily
    model_name: str

    def load(self, status_callback: StatusCallback | None = None) -> None: ...

    def warmup(self, status_callback: StatusCallback | None = None) -> None: ...

    def generate(
        self,
        job: ResolvedGenerationJob,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult: ...


# ---------------------------------------------------------------------------
# Normalization and validation
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def round_dimension(value: int, multiple: int = DIMENSION_MULTIPLE) -> int:
    """Nearest multiple of 8 (or ``multiple``), ties rounding up (516 -> 520, 1001 -> 1000)."""
    return (value + multiple // 2) // multiple * multiple


def min_img2img_steps(strength: float) -> int:
    """Smallest step count for which ``int(steps * strength)`` is at least 1."""
    steps = max(1, math.ceil(1 / strength))
    while int(steps * strength) < 1:  # guard against float rounding at the boundary
        steps += 1
    return steps


def normalize_request(
    req: GenerationRequest, family: ModelFamily
) -> tuple[GenerationRequest, tuple[str, ...]]:
    """Apply silent backend corrections; return the corrected request and log notes.

    - width/height are rounded to the nearest multiple of 8 (16 for Z-Image; range is
      checked afterwards);
    - img2img steps are raised so at least one denoising step runs (Diffusers runs
      ``int(steps * strength)`` steps and errors on zero).

    Values of the wrong type are left untouched for ``validate_request`` to reject.
    """
    changes: dict[str, int] = {}
    notes: list[str] = []
    multiple = FAMILIES[family].dimension_multiple

    for name in ("width", "height"):
        value = getattr(req, name)
        if _is_int(value) and value % multiple:
            changes[name] = round_dimension(value, multiple)
            notes.append(f"{name} {value} rounded to {changes[name]} (multiple of {multiple})")

    if (
        req.input_image is not None
        and _is_int(req.steps)
        and req.steps >= 1
        and _is_number(req.strength)
        and 0 < req.strength <= 1
        and int(req.steps * req.strength) < 1
        and min_img2img_steps(req.strength) <= MAX_STEPS
    ):
        changes["steps"] = min_img2img_steps(req.strength)
        notes.append(
            f"steps raised from {req.steps} to {changes['steps']} so strength "
            f"{req.strength} runs at least 1 denoising step"
        )

    return (replace(req, **changes) if changes else req), tuple(notes)


def validate_request(req: GenerationRequest, family: ModelFamily) -> None:
    """Backend-side validation for ``family``; raises ValidationError with a user-facing message.

    Expects a request that has already been through ``normalize_request``.
    """
    spec = FAMILIES[family]
    for name in ("prompt", "negative_prompt"):
        text = getattr(req, name)
        if not isinstance(text, str):
            raise ValidationError(f"{name} must be a string")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValidationError(f"{name} is longer than {MAX_PROMPT_CHARS} characters")
        try:
            validate_template(text)
        except PromptSyntaxError as exc:
            raise ValidationError(f"{name}: {exc}") from exc

    for name in ("width", "height"):
        value = getattr(req, name)
        if not _is_int(value):
            raise ValidationError(f"{name} must be an integer")
        if not MIN_DIMENSION <= value <= MAX_DIMENSION:
            raise ValidationError(f"{name} must be between {MIN_DIMENSION} and {MAX_DIMENSION}")
        if value % spec.dimension_multiple:
            raise ValidationError(f"{name} must be a multiple of {spec.dimension_multiple}")

    if not _is_int(req.steps) or not 1 <= req.steps <= MAX_STEPS:
        raise ValidationError(f"steps must be an integer between 1 and {MAX_STEPS}")

    if not _is_number(req.guidance_scale) or not 0 <= req.guidance_scale <= MAX_GUIDANCE:
        raise ValidationError(f"guidance scale must be between 0 and {MAX_GUIDANCE}")

    if req.sampler not in spec.samplers:
        raise ValidationError(f"sampler {req.sampler!r} is not available for {family}")

    if not _is_int(req.num_images) or not 1 <= req.num_images <= MAX_IMAGES:
        raise ValidationError(f"image count must be between 1 and {MAX_IMAGES}")

    if not _is_int(req.seed) or not (req.seed == -1 or 0 <= req.seed <= MAX_SEED):
        raise ValidationError(f"seed must be -1 or an integer between 0 and {MAX_SEED}")
    if req.seed != -1 and req.seed + req.num_images - 1 > MAX_SEED:
        raise ValidationError(f"seed sequence would exceed the maximum seed {MAX_SEED}")

    if req.input_image is None:
        if req.strength is not None:
            raise ValidationError("strength applies only to image-to-image requests")
    else:
        if not _is_number(req.strength) or not 0 < req.strength <= 1:
            raise ValidationError("img2img strength must be greater than 0 and at most 1")
        # Diffusers img2img runs int(steps * strength) denoising steps; zero is an error.
        if int(req.steps * req.strength) < 1:
            raise ValidationError(
                f"strength {req.strength} needs more than {MAX_STEPS} steps to denoise"
            )


# ---------------------------------------------------------------------------
# Seeds and job resolution
# ---------------------------------------------------------------------------


def resolve_seeds(seed: int, count: int, rng: random.Random | None = None) -> tuple[int, ...]:
    """``-1`` -> ``count`` unique random seeds; otherwise ``seed, seed+1, ...``."""
    if seed != -1:
        return tuple(seed + i for i in range(count))
    rng = rng or random.SystemRandom()
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = rng.randint(0, MAX_SEED)
        if candidate not in seeds:
            seeds.append(candidate)
    return tuple(seeds)


def resolve_job(
    req: GenerationRequest,
    job_id: int = 0,
    rng: random.Random | None = None,
    *,
    family: ModelFamily,
) -> ResolvedGenerationJob:
    """Normalize, validate, and resolve concrete seeds and dynamic prompts.

    Done at enqueue time. Each image's dynamic choices come from its own seed, so a
    fixed-seed job (123, 124, ...) resolves the same prompts every time it is re-run.
    """
    req, corrections = normalize_request(req, family)
    validate_request(req, family)
    weights = FAMILIES[family].prompt_weights
    images = []
    for index, seed in enumerate(resolve_seeds(req.seed, req.num_images, rng)):
        prompts = resolve_prompt_pair(req.prompt, req.negative_prompt, seed)
        if not weights:
            prompts = tuple(map(strip_weights, prompts))
        images.append(ResolvedImageSpec(index, seed, *prompts))
    if not weights and any(strip_weights(t) != t for t in (req.prompt, req.negative_prompt)):
        corrections += (f"prompt weights like (text:1.2) are ignored for {family}",)
    if req.negative_prompt.strip() and req.guidance_scale == 0:
        corrections += ("negative prompt is ignored at CFG 0",)
    return ResolvedGenerationJob(
        request=req, images=tuple(images), job_id=job_id, corrections=corrections
    )


def describe_request(job: ResolvedGenerationJob) -> list[str]:
    """Job summary: a settings line (no prompt text), then per-image prompts if dynamic."""
    req = job.request
    strength = f" · strength {req.strength}" if req.input_image else ""
    lines = [
        f"{req.mode} · {SAMPLER_LABELS[req.sampler]} · {req.steps} steps · "
        f"{req.width}x{req.height} · CFG {req.guidance_scale}{strength} · "
        f"{req.num_images} image(s) · seeds {', '.join(map(str, job.seeds))}"
    ]
    if any(spec.prompt != req.prompt for spec in job.images):
        lines += [f"image {s.index + 1} (seed {s.seed}): {s.prompt}" for s in job.images]
    return lines


# ---------------------------------------------------------------------------
# Input-image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(source: Path | Image.Image, width: int, height: int) -> Image.Image:
    """EXIF-correct, RGB-convert, cover-resize (LANCZOS) and center-crop to width x height.

    Aspect ratio is always preserved; overflow is cropped, never stretched.
    """
    if isinstance(source, Image.Image):
        image = source
    else:
        with Image.open(source) as opened:
            opened.load()
            image = opened.copy()
    image = ImageOps.exif_transpose(image).convert("RGB")

    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w = max(width, round(src_w * scale))
    new_h = max(height, round(src_h * scale))
    if (new_w, new_h) != (src_w, src_h):
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return image.crop((left, top, left + width, top + height))


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------


def output_filename(job_id: int, index: int, seed: int, now: datetime | None = None) -> str:
    """e.g. ``20261003_162455_job0007_img01_seed123.png`` (``index`` is 0-based)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_job{job_id:04d}_img{index + 1:02d}_seed{seed}.png"


def collision_safe_path(directory: Path, filename: str) -> Path:
    """Return ``directory/filename``, adding ``_1``, ``_2``... before the suffix if taken."""
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def denoising_steps(req: GenerationRequest) -> int:
    """Number of denoising steps the pipeline actually runs (img2img skips the early ones)."""
    if req.input_image is None:
        return req.steps
    return int(req.steps * req.strength)


# ---------------------------------------------------------------------------
# Micro-batching with OOM fallback (SPEC §8) — shared by mock and real backends
# ---------------------------------------------------------------------------

T = TypeVar("T")
R = TypeVar("R")
BatchKey = tuple[str, int, int]  # (mode, width, height)


class MicroBatcher:
    """Runs a job's images in as few batches as VRAM allows.

    Tries the whole remaining set (capped by any limit learned earlier for this key).
    On OOM the batch size halves and retries; after an OOM the size that then works is
    remembered as the limit for that (mode, width, height) for the rest of the session.
    """

    def __init__(self) -> None:
        self.limits: dict[BatchKey, int] = {}

    def run(
        self,
        key: BatchKey,
        items: Sequence[T],
        run_batch: Callable[[Sequence[T], Callable[[int, int], None]], Sequence[R]],
        is_oom: Callable[[BaseException], bool],
        progress_callback: ProgressCallback | None = None,
        on_oom: Callable[[int, int], None] | None = None,
        on_batch_done: Callable[[Sequence[R]], None] | None = None,
    ) -> list[R]:
        """``run_batch(batch, on_step)`` must call ``on_step(step, total_steps)`` per step.

        ``on_oom(failed_size, next_size)`` is called after each OOM (e.g. to free cache).
        ``on_batch_done(results)`` is called after each successful batch.
        """
        total = len(items)
        results: list[R] = []
        size = min(self.limits.get(key, total), total)  # working batch size
        had_oom = False
        batch_number = 0

        while len(results) < total:
            done = len(results)
            n = min(size, total - done)  # the last batch may be a smaller remainder
            batch = items[done : done + n]
            batch_number += 1
            batch_count = batch_number + math.ceil((total - done - n) / size)
            prefix = f"Batch {batch_number}/{batch_count} · " if batch_count > 1 else ""

            def on_step(step: int, total_steps: int, done=done, n=n, prefix=prefix) -> None:
                if progress_callback is None:
                    return
                fraction = (done + n * step / max(total_steps, 1)) / total
                progress_callback(fraction, f"{prefix}denoising {step}/{total_steps}")

            try:
                batch_results = run_batch(batch, on_step)
            except Exception as exc:
                if not is_oom(exc) or n == 1:
                    raise
                size = max(1, n // 2)
                had_oom = True
                batch_number -= 1
                if on_oom is not None:
                    on_oom(n, size)
                continue

            # Only a full-size success proves the working size; a remainder batch doesn't.
            if had_oom and n == size:
                self.limits[key] = min(self.limits.get(key, size), size)
            results.extend(batch_results)
            if on_batch_done is not None:
                on_batch_done(batch_results)

        return results


class _BatchedBackend:
    """Shared job loop: preprocess the input, micro-batch with OOM fallback, save PNGs.

    Subclasses provide ``loaded``, ``batcher`` and ``_render(req, batch, source, on_step)``
    returning (or yielding) one PIL image per spec, plus ``_is_oom`` / ``_on_oom``.
    """

    def generate(
        self,
        job: ResolvedGenerationJob,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        if not self.loaded:
            raise GenerationError("Backend is not loaded")
        req = job.request
        started, now = time.monotonic(), datetime.now()
        source = (
            preprocess_image(req.input_image, req.width, req.height) if req.input_image else None
        )
        written: list[Path] = []

        def run_batch(batch: Sequence[ResolvedImageSpec], on_step) -> list[Path]:
            for spec, image in zip(batch, self._render(req, batch, source, on_step), strict=True):
                name = output_filename(job.job_id, spec.index, spec.seed, now)
                written.append(collision_safe_path(output_dir, name))
                save_png(image, written[-1])
            return written[-len(batch) :]

        try:
            paths = self.batcher.run(
                (req.mode, req.width, req.height),
                job.images,
                run_batch,
                is_oom=self._is_oom,
                progress_callback=progress_callback,
                on_oom=self._on_oom,
            )
        except Exception as exc:
            raise GenerationError(f"{type(exc).__name__}: {exc}", written) from exc
        return GenerationResult(
            output_paths=tuple(paths),
            seeds=job.seeds,
            resolved_prompts=tuple(spec.prompt for spec in job.images),
            elapsed_seconds=time.monotonic() - started,
        )

    def _on_oom(self, failed: int, next_size: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock backend (SPEC §8): no torch, no diffusers, no network, no GPU
# ---------------------------------------------------------------------------


class MockOutOfMemory(RuntimeError):
    """Simulated CUDA OOM raised when a mock batch exceeds ``max_batch``."""


class MockBackend(_BatchedBackend):
    """Implements the backend contract with placeholder PNGs drawn by Pillow.

    ``step_seconds`` / ``load_seconds`` add delays so queue/progress UI is observable.
    ``max_batch`` simulates VRAM limits (batches larger than it raise MockOutOfMemory).
    ``fail_load`` / ``fail_after_images`` simulate fatal load / mid-job errors.
    """

    def __init__(
        self,
        family: ModelFamily,
        model_name: str = "mock.safetensors",
        *,
        load_seconds: float = 0.0,
        step_seconds: float = 0.0,
        max_batch: int | None = None,
        fail_load: bool = False,
        fail_after_images: int | None = None,
    ) -> None:
        self.family = check_family(family)
        self.model_name = model_name
        self.load_seconds = load_seconds
        self.step_seconds = step_seconds
        self.max_batch = max_batch
        self.fail_load = fail_load
        self.fail_after_images = fail_after_images
        self.batcher = MicroBatcher()
        self.batch_sizes: list[int] = []  # every attempted batch size, for tests
        self.images_drawn = 0
        self.loaded = False
        self.load_count = 0

    def load(self, status_callback: StatusCallback | None = None) -> None:
        status = status_callback or (lambda state, message: None)
        status("loading", "Loading checkpoint (mock)…")
        self.load_count += 1
        time.sleep(self.load_seconds)
        if self.fail_load:
            raise RuntimeError("Mock checkpoint failed to load")
        status("optimizing", "Optimizing (mock)…")
        time.sleep(self.load_seconds)
        self.loaded = True

    def warmup(self, status_callback: StatusCallback | None = None) -> None:
        (status_callback or (lambda state, message: None))("warming", "Warm-up 1/1…")
        time.sleep(self.load_seconds)

    def _is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, MockOutOfMemory)

    def _render(self, req, batch, source, on_step):
        self.batch_sizes.append(len(batch))
        if self.max_batch is not None and len(batch) > self.max_batch:
            raise MockOutOfMemory(f"mock OOM at batch size {len(batch)}")
        total_steps = denoising_steps(req)
        for step in range(1, total_steps + 1):
            time.sleep(self.step_seconds)
            on_step(step, total_steps)
        for spec in batch:  # a generator, so a simulated failure leaves earlier PNGs written
            if self.fail_after_images is not None and self.images_drawn >= self.fail_after_images:
                raise RuntimeError("Simulated mock generation failure")
            self.images_drawn += 1
            yield self._draw(spec, req, source)

    def _draw(
        self, spec: ResolvedImageSpec, req: GenerationRequest, source: Image.Image | None
    ) -> Image.Image:
        # Colour depends on seed + prompts, but no prompt text is ever drawn (privacy).
        rng = random.Random(f"{spec.seed}|{spec.prompt}|{spec.negative_prompt}")
        color = tuple(rng.randrange(40, 216) for _ in range(3))
        if source is None:
            image = Image.new("RGB", (req.width, req.height), color)
        else:
            image = Image.blend(source, Image.new("RGB", source.size, color), req.strength * 0.6)
        # No batch index: like the real backend, an image depends only on seed/prompt/mode.
        text = f"MOCK {self.family.upper()} · {req.mode}\nseed {spec.seed}"
        font = ImageFont.load_default(size=max(12, req.width // 24))
        ImageDraw.Draw(image).multiline_text(
            (16, 16), text, fill="white", font=font, spacing=6, stroke_width=2, stroke_fill="black"
        )
        return image


def save_png(image: Image.Image, path: Path) -> None:
    """Save as PNG with no generation metadata (no text chunks, no EXIF; SPEC §7)."""
    # Copy pixels into a fresh image so no ``info`` (text, EXIF, ICC) carries over.
    clean = Image.new(image.mode, image.size)
    clean.paste(image)
    clean.save(path, format="PNG")


# ---------------------------------------------------------------------------
# Real Diffusers backend (SPEC §4, §8). torch/diffusers imported lazily.
# ---------------------------------------------------------------------------

# Optimization profiles compared in Phase 8 (SPEC §8, §13). "baseline" = FP16 + SDPA.
OPTIMIZATION_PROFILES: tuple[str, ...] = ("baseline", "compile", "compile-max")
DEFAULT_OPTIMIZATION = "baseline"  # until Phase 8 L4 benchmarks pick a winner
WARMUP_STEPS = 3

# Z-Image single files hold only the transformer; text encoder (Qwen3-4B), VAE, tokenizer,
# scheduler and configs come from the official Diffusers repo (~7.7 GiB, cached by the Hub).
ZIMAGE_COMPONENTS = "Tongyi-MAI/Z-Image-Turbo"
_ZIMAGE_KEYS = (
    "layers.0.adaLN_modulation.0.weight",
    "model.diffusion_model.layers.0.adaLN_modulation.0.weight",
)
_FP8_DTYPES = {"F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
SCALED_FP8 = "scaled_fp8"
# ComfyUI quantized layers: X.weight (FP8) + X.weight_scale (F32) + X.comfy_quant (U8 JSON
# config, e.g. {"format": "float8_e4m3fn"}) and optionally X.input_scale (activations only).
_COMFY_QUANT_EXTRAS = (".comfy_quant", ".input_scale")


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Tensor name -> {dtype, shape, ...} from a .safetensors header (no tensor data read)."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        if not 2 <= size <= 100 * 2**20:
            raise ValueError(f"{path.name} is not a valid .safetensors file")
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def zimage_weight_format(path: Path) -> str | None:
    """Weight format of a Z-Image transformer file, from its header.

    ``None``: BF16/FP16/FP32 weights. ``"float8_e4m3fn"`` / ``"float8_e5m2"``: plain FP8
    (weights cast straight to FP8, same names as the BF16 release). ``SCALED_FP8``: ComfyUI's
    quantized FP8 (FP8 weight x F32 ``weight_scale`` per layer). Other quantized formats
    (int8, NVFP4, other scale layouts) need their own loaders and are rejected.
    """
    header = read_safetensors_header(path)
    if not any(key in header for key in _ZIMAGE_KEYS):
        raise ValueError(f"{path.name} is not a Z-Image transformer checkpoint")
    if "all_x_embedder.2-1.weight" in header:  # the single-file converter expects ComfyUI names
        raise ValueError(f"{path.name} uses Diffusers tensor names; use a ComfyUI-layout file")
    scaled = {key.removesuffix("_scale") for key in header if key.endswith(".weight_scale")}
    weights = {
        key: info["dtype"]
        for key, info in header.items()
        if not key.endswith((*_COMFY_QUANT_EXTRAS, ".weight_scale"))
    }
    if any("scale" in key.rsplit(".", 1)[-1] for key in weights):
        raise ValueError(f"{path.name} uses an unsupported quantization; use BF16 or FP8")
    dtypes = set(weights.values())
    fp8 = {_FP8_DTYPES[d] for d in dtypes if d in _FP8_DTYPES}
    unsupported = dtypes - set(_FP8_DTYPES) - {"BF16", "F16", "F32"}
    if unsupported or len(fp8) > 1 or any(weights.get(w) not in _FP8_DTYPES for w in scaled):
        raise ValueError(f"{path.name}: unsupported weight types {', '.join(sorted(dtypes))}")
    if scaled:
        return SCALED_FP8
    return fp8.pop() if fp8 else None


def describe_weight_format(fmt: str | None, compute: str = "bfloat16") -> str:
    if fmt == SCALED_FP8:
        return f"scaled FP8, dequantized to {compute}"
    return f"{fmt} storage, {compute} compute" if fmt else compute


def dequantize_scaled_fp8(path: Path, dtype: Any) -> dict[str, Any]:
    """ComfyUI scaled-FP8 file -> plain state dict: ``weight x weight_scale`` in ``dtype``.

    The scale is a scalar or one value per output row (broadcasts either way).
    """
    from safetensors.torch import load_file

    state = load_file(str(path))
    for key in [k for k in state if k.endswith(".weight_scale")]:
        weight = key.removesuffix("_scale")
        state[weight] = (state[weight].float() * state.pop(key)).to(dtype)
    return {k: v for k, v in state.items() if not k.endswith(_COMFY_QUANT_EXTRAS)}


def configure_cuda_allocator() -> None:
    """Set the allocator config; only effective if called before torch is imported."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def make_scheduler(sampler: str, base_config: Any) -> Any:
    """Build a fresh scheduler for ``sampler`` from the checkpoint's scheduler config."""
    import diffusers

    class_name, overrides = SAMPLER_SCHEDULERS[sampler]
    return getattr(diffusers, class_name).from_config(base_config, **overrides)


class DiffusersBackend(_BatchedBackend):
    """Single-file SD1.5/SDXL checkpoint or Z-Image transformer kept resident on the GPU.

    ``pipeline_loader`` replaces ``from_single_file`` (tests inject tiny random
    pipelines); ``device``/``dtype`` default to the Colab L4 fast path (dtype per family).
    ``zimage_components`` is where Z-Image's other components come from (a Hub repo id
    or a local Diffusers folder).
    """

    def __init__(
        self,
        family: ModelFamily,
        model_path: Path | str,
        *,
        device: str = "cuda",
        dtype: str | None = None,
        optimization: str = DEFAULT_OPTIMIZATION,
        warmup_steps: int = WARMUP_STEPS,
        warmup_size: tuple[int, int] | None = None,
        log: Callable[[str], None] | None = None,
        pipeline_loader: Callable[[], Any] | None = None,
        zimage_components: str = ZIMAGE_COMPONENTS,
    ) -> None:
        if optimization not in OPTIMIZATION_PROFILES:
            raise ValueError(f"optimization must be one of {', '.join(OPTIMIZATION_PROFILES)}")
        self.family = check_family(family)
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        self.device = device
        self.spec = FAMILIES[self.family]
        self.dtype_name = dtype or self.spec.dtype
        self.zimage_components = zimage_components
        self.optimization = optimization
        self.active_optimization = "baseline"
        self.warmup_steps = warmup_steps
        self.warmup_size = warmup_size or (self.spec.defaults.width, self.spec.defaults.height)
        self.log = log or (lambda message: None)
        self.pipeline_loader = pipeline_loader
        self.batcher = MicroBatcher()
        self.txt2img: Any = None
        self.img2img: Any = None
        self.base_scheduler_config: Any = None
        self._original_modules: dict[str, Any] = {}
        self._seen_shapes: set[BatchKey] = set()
        self.load_count = 0

    @property
    def loaded(self) -> bool:
        return self.txt2img is not None

    # -- loading -------------------------------------------------------------

    def load(self, status_callback: StatusCallback | None = None) -> None:
        status = status_callback or (lambda state, message: None)
        configure_cuda_allocator()
        status("loading", "Loading checkpoint…")
        self.load_count += 1
        started = time.monotonic()
        pipe = self.pipeline_loader() if self.pipeline_loader else self._load_single_file()
        pipe.set_progress_bar_config(disable=True)
        self.txt2img = pipe
        self.img2img = self._img2img_from(pipe)
        self.base_scheduler_config = pipe.scheduler.config
        self.log(f"Loaded {self.model_name} ({self.family}) in {time.monotonic() - started:.1f}s")
        if self.optimization != "baseline":
            status("optimizing", "Applying optimizations…")
            self._apply_optimization(self.optimization)

    def _load_single_file(self) -> Any:
        import diffusers
        import torch

        if not self.model_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.model_path}")
        dtype = getattr(torch, self.dtype_name)
        if self.family == "sd15":
            pipe = diffusers.StableDiffusionPipeline.from_single_file(
                str(self.model_path),
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        elif self.family == "sdxl":
            pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
                str(self.model_path), torch_dtype=dtype, add_watermarker=False
            )
        else:
            pipe = self._load_zimage(dtype)
        return pipe.to(self.device)

    def _load_zimage(self, dtype: Any) -> Any:
        import diffusers
        import torch

        path, fmt = self.model_path, zimage_weight_format(self.model_path)
        # Scaled FP8 is dequantized once, exactly, to BF16 (re-casting to plain FP8 would
        # quantize twice), so it needs BF16 VRAM.
        source = dequantize_scaled_fp8(path, dtype) if fmt == SCALED_FP8 else str(path)
        transformer = diffusers.ZImageTransformer2DModel.from_single_file(
            source, config=self.zimage_components, subfolder="transformer", torch_dtype=dtype
        )
        if fmt in _FP8_DTYPES.values():
            # Keep weights in FP8 (half the VRAM), upcast per layer to compute in BF16. Exact
            # for a plain FP8 file; saves memory, not time. Precision-sensitive layers stay BF16.
            transformer.enable_layerwise_casting(
                storage_dtype=getattr(torch, fmt), compute_dtype=dtype
            )
        self.log(
            f"Transformer weights: {describe_weight_format(fmt, self.dtype_name)}; loading text "
            f"encoder + VAE from {self.zimage_components} (first run downloads ~7.7 GiB)"
        )
        return diffusers.ZImagePipeline.from_pretrained(
            self.zimage_components, transformer=transformer, torch_dtype=dtype
        )

    def _img2img_from(self, pipe: Any) -> Any:
        """Img2img pipeline sharing the already-loaded components (no second weight copy).

        ``from_pipe`` casts every shared module to its ``dtype`` argument (default float32!),
        so it gets the pipeline's own dtype. Z-Image is built from the components directly:
        any cast would also undo FP8 storage.
        """
        import diffusers

        if self.family == "sd15":
            return diffusers.StableDiffusionImg2ImgPipeline.from_pipe(
                pipe, safety_checker=None, requires_safety_checker=False, torch_dtype=pipe.dtype
            )
        if self.family == "zimage":
            return diffusers.ZImageImg2ImgPipeline(**pipe.components)
        img2img = diffusers.StableDiffusionXLImg2ImgPipeline.from_pipe(pipe, torch_dtype=pipe.dtype)
        img2img.watermark = None  # SPEC §4: never watermark
        return img2img

    # -- optional acceleration -----------------------------------------------

    def _set_module(self, name: str, module: Any) -> None:
        for pipe in (self.txt2img, self.img2img):
            setattr(pipe, name, module)

    def _apply_optimization(self, profile: str) -> None:
        """channels_last + torch.compile; any failure falls back to the baseline."""
        import torch

        name = self.spec.denoiser
        self._original_modules = {name: getattr(self.txt2img, name), "vae": self.txt2img.vae}
        try:
            mode = "max-autotune" if profile == "compile-max" else "reduce-overhead"
            denoiser = self._original_modules[name]
            if name == "unet":
                denoiser = denoiser.to(memory_format=torch.channels_last)
            self._set_module(name, torch.compile(denoiser, mode=mode, fullgraph=True))
            if profile == "compile-max":
                vae = self.txt2img.vae.to(memory_format=torch.channels_last)
                vae.decode = torch.compile(vae.decode, mode=mode, fullgraph=True)
            self.active_optimization = profile
            self.log(f"Optimization: {profile} (torch.compile {mode}); compiles on first use")
        except Exception as exc:  # noqa: BLE001 — any optional-acceleration failure
            self._revert_optimization(f"{type(exc).__name__}: {exc}")

    def _revert_optimization(self, reason: str) -> None:
        for name, module in self._original_modules.items():
            if name == "vae" and "decode" in vars(module):
                del module.decode  # drop the compiled instance attribute
            self._set_module(name, module)
        self.active_optimization = "baseline"
        self.log(f"Optimization failed, falling back to baseline: {reason}")

    # -- diagnostics -----------------------------------------------------------

    def reset_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def memory_stats(self) -> dict[str, float] | None:
        """Peak CUDA memory since the last reset, in GiB (None without CUDA)."""
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        }

    # -- warm-up -------------------------------------------------------------

    def warmup(self, status_callback: StatusCallback | None = None) -> None:
        """One batch-1 generation at the family default size; no output file."""
        status = status_callback or (lambda state, message: None)
        status("warming", "Warm-up 1/1…")
        started = time.monotonic()
        try:
            self._warmup_once()
        except Exception as exc:  # noqa: BLE001
            if self.active_optimization == "baseline":
                raise
            self._revert_optimization(f"warm-up failed: {type(exc).__name__}: {exc}")
            self._warmup_once()
        self.log(f"Warm-up complete in {time.monotonic() - started:.1f}s")

    def _warmup_once(self) -> None:
        d = self.spec.defaults
        width, height = self.warmup_size
        req = GenerationRequest(
            "warm-up", "", width, height, self.warmup_steps, d.guidance_scale,
            d.sampler, 0, 1,
        )  # fmt: skip
        spec = ResolvedImageSpec(0, 0, req.prompt, req.negative_prompt)
        self._run_pipeline(req, [spec], None, lambda step, total: None)
        self._seen_shapes.add((req.mode, req.width, req.height))

    # -- generation (job loop in _BatchedBackend) ------------------------------

    def _is_oom(self, exc: BaseException) -> bool:
        import torch

        return isinstance(exc, torch.OutOfMemoryError)

    def _on_oom(self, failed: int, next_size: int) -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log(f"CUDA out of memory at batch {failed}; retrying with batch {next_size}")

    def _render(self, req, batch, source, on_step):
        key: BatchKey = (req.mode, req.width, req.height)
        if self.active_optimization != "baseline" and key not in self._seen_shapes:
            self.log(
                f"Optimizing new tensor shape {req.width}x{req.height}; "
                "first generation at this size may be slower."
            )
        self._seen_shapes.add(key)
        return self._run_pipeline(req, batch, source, on_step)

    def _run_pipeline(
        self,
        req: GenerationRequest,
        batch: Sequence[ResolvedImageSpec],
        source: Image.Image | None,
        on_step: Callable[[int, int], None],
    ) -> list[Image.Image]:
        import torch

        from src.prompting import encode_prompt_batch

        pipe = self.txt2img if source is None else self.img2img
        pipe.scheduler = make_scheduler(req.sampler, self.base_scheduler_config)
        prompts = [spec.prompt for spec in batch]
        negatives = [spec.negative_prompt for spec in batch]
        if self.spec.prompt_weights:
            embeds = encode_prompt_batch(pipe, self.family, prompts, negatives)
        else:  # e.g. Z-Image: Qwen3 chat-template encoding stays in the pipeline
            embeds = {"prompt": prompts, "negative_prompt": negatives}
        # One generator per image: an image depends only on its own seed (SPEC §7).
        generators = [torch.Generator(device=self.device).manual_seed(s.seed) for s in batch]

        def callback(pipeline: Any, step_index: int, _timestep: Any, kwargs: dict) -> dict:
            on_step(step_index + 1, pipeline.num_timesteps)
            return kwargs

        call_kwargs: dict[str, Any] = dict(
            embeds,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=generators,
            callback_on_step_end=callback,
            output_type="pil",
        )
        if source is None:
            call_kwargs.update(width=req.width, height=req.height)
        else:
            call_kwargs.update(image=[source] * len(batch), strength=req.strength)
        with torch.inference_mode():
            return pipe(**call_kwargs).images
