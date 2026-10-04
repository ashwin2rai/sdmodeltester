"""Request/result types, pure-Python job resolution, and generation backends.

This module must stay importable without torch/diffusers. The real Diffusers
backend (Phase 3) imports them lazily inside its methods.
"""

from __future__ import annotations

import gc
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

from src.prompting import PromptSyntaxError, resolve_prompt_pair, validate_template

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

ModelFamily = Literal["sd15", "sdxl"]
MODEL_FAMILIES: tuple[str, ...] = get_args(ModelFamily)

Mode = Literal["txt2img", "img2img"]

# Public sampler IDs are application-owned (SPEC §8); never Diffusers class names.
SAMPLERS: tuple[tuple[str, str], ...] = (
    ("dpmpp_2m_karras", "DPM++ 2M Karras"),
    ("dpmpp_2m_sde_karras", "DPM++ 2M SDE Karras"),
    ("euler", "Euler"),
    ("euler_a", "Euler a"),
    ("heun", "Heun"),
    ("dpm2_karras", "DPM2 Karras"),
)
SAMPLER_IDS: tuple[str, ...] = tuple(sid for sid, _ in SAMPLERS)
SAMPLER_LABELS: dict[str, str] = dict(SAMPLERS)

# Sampler ID -> (diffusers scheduler class name, from_config overrides). SPEC §8.
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
}

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


FAMILY_DEFAULTS: dict[str, FamilyDefaults] = {
    "sd15": FamilyDefaults(512, 512, 25, 7.5, "dpmpp_2m_sde_karras", -1, 1, 0.6),
    "sdxl": FamilyDefaults(1024, 1024, 25, 5.0, "dpmpp_2m_sde_karras", -1, 1, 0.6),
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


def round_dimension(value: int) -> int:
    """Nearest multiple of 8, ties rounding up (516 -> 520, 1001 -> 1000)."""
    half = DIMENSION_MULTIPLE // 2
    return (value + half) // DIMENSION_MULTIPLE * DIMENSION_MULTIPLE


def min_img2img_steps(strength: float) -> int:
    """Smallest step count for which ``int(steps * strength)`` is at least 1."""
    steps = max(1, math.ceil(1 / strength))
    while int(steps * strength) < 1:  # guard against float rounding at the boundary
        steps += 1
    return steps


def normalize_request(req: GenerationRequest) -> tuple[GenerationRequest, tuple[str, ...]]:
    """Apply silent backend corrections; return the corrected request and log notes.

    - width/height are rounded to the nearest multiple of 8 (range is checked afterwards);
    - img2img steps are raised so at least one denoising step runs (Diffusers runs
      ``int(steps * strength)`` steps and errors on zero).

    Values of the wrong type are left untouched for ``validate_request`` to reject.
    """
    changes: dict[str, int] = {}
    notes: list[str] = []

    for name in ("width", "height"):
        value = getattr(req, name)
        if _is_int(value) and value % DIMENSION_MULTIPLE:
            changes[name] = round_dimension(value)
            notes.append(f"{name} {value} rounded to {changes[name]} (multiple of 8)")

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


def validate_request(req: GenerationRequest) -> None:
    """Backend-side validation; raises ValidationError with a user-facing message.

    Expects a request that has already been through ``normalize_request``.
    """
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
        if value % DIMENSION_MULTIPLE:
            raise ValidationError(f"{name} must be a multiple of {DIMENSION_MULTIPLE}")

    if not _is_int(req.steps) or not 1 <= req.steps <= MAX_STEPS:
        raise ValidationError(f"steps must be an integer between 1 and {MAX_STEPS}")

    if not _is_number(req.guidance_scale) or not 0 <= req.guidance_scale <= MAX_GUIDANCE:
        raise ValidationError(f"guidance scale must be between 0 and {MAX_GUIDANCE}")

    if req.sampler not in SAMPLER_IDS:
        raise ValidationError(f"unknown sampler {req.sampler!r}")

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
    req: GenerationRequest, job_id: int = 0, rng: random.Random | None = None
) -> ResolvedGenerationJob:
    """Normalize, validate, and resolve concrete seeds and dynamic prompts.

    Done at enqueue time. Each image's dynamic choices come from its own seed, so a
    fixed-seed job (123, 124, ...) resolves the same prompts every time it is re-run.
    """
    req, corrections = normalize_request(req)
    validate_request(req)
    images = []
    for index, seed in enumerate(resolve_seeds(req.seed, req.num_images, rng)):
        prompt, negative = resolve_prompt_pair(req.prompt, req.negative_prompt, seed)
        images.append(ResolvedImageSpec(index, seed, prompt, negative))
    return ResolvedGenerationJob(
        request=req, images=tuple(images), job_id=job_id, corrections=corrections
    )


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
# Micro-batching with OOM fallback (SPEC §14.3) — shared by mock and real backends
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


# ---------------------------------------------------------------------------
# Mock backend (SPEC §15): no torch, no diffusers, no network, no GPU
# ---------------------------------------------------------------------------


class MockOutOfMemory(RuntimeError):
    """Simulated CUDA OOM raised when a mock batch exceeds ``max_batch``."""


class MockBackend:
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
        self.batch_sizes: list[int] = []  # every attempted batch size, for tests/logs
        self.loaded = False

    def load(self, status_callback: StatusCallback | None = None) -> None:
        status = status_callback or (lambda state, message: None)
        status("loading", "Loading checkpoint (mock)…")
        time.sleep(self.load_seconds)
        if self.fail_load:
            raise RuntimeError("Mock checkpoint failed to load")
        status("optimizing", "Optimizing (mock)…")
        time.sleep(self.load_seconds)
        self.loaded = True

    def warmup(self, status_callback: StatusCallback | None = None) -> None:
        status = status_callback or (lambda state, message: None)
        status("warming", "Warm-up 1/1…")
        time.sleep(self.load_seconds)

    def generate(
        self,
        job: ResolvedGenerationJob,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        if not self.loaded:
            raise GenerationError("Backend is not loaded")
        req = job.request
        started = time.monotonic()
        now = datetime.now()
        source = (
            preprocess_image(req.input_image, req.width, req.height)
            if req.input_image is not None
            else None
        )
        total_steps = denoising_steps(req)
        written: list[Path] = []

        def run_batch(
            batch: Sequence[ResolvedImageSpec], on_step: Callable[[int, int], None]
        ) -> list[Path]:
            self.batch_sizes.append(len(batch))
            if self.max_batch is not None and len(batch) > self.max_batch:
                raise MockOutOfMemory(f"mock OOM at batch size {len(batch)}")
            for step in range(1, total_steps + 1):
                time.sleep(self.step_seconds)
                on_step(step, total_steps)
            paths = []
            for spec in batch:
                if self.fail_after_images is not None and len(written) >= self.fail_after_images:
                    raise RuntimeError("Simulated mock generation failure")
                image = self._draw(spec, req, source)
                path = collision_safe_path(
                    output_dir, output_filename(job.job_id, spec.index, spec.seed, now)
                )
                save_png(image, path)
                written.append(path)
                paths.append(path)
            return paths

        try:
            paths = self.batcher.run(
                (req.mode, req.width, req.height),
                job.images,
                run_batch,
                is_oom=lambda exc: isinstance(exc, MockOutOfMemory),
                progress_callback=progress_callback,
            )
        except Exception as exc:
            raise GenerationError(str(exc), written) from exc

        return GenerationResult(
            output_paths=tuple(paths),
            seeds=job.seeds,
            resolved_prompts=tuple(spec.prompt for spec in job.images),
            elapsed_seconds=time.monotonic() - started,
        )

    def _draw(
        self, spec: ResolvedImageSpec, req: GenerationRequest, source: Image.Image | None
    ) -> Image.Image:
        color = tuple(random.Random(spec.seed).randrange(40, 216) for _ in range(3))
        if source is None:
            image = Image.new("RGB", (req.width, req.height), color)
        else:
            tint = Image.new("RGB", source.size, color)
            image = Image.blend(source, tint, req.strength * 0.6)

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(12, req.width // 24))
        lines = [
            f"MOCK {self.family.upper()} · {req.mode}",
            f"#{spec.index + 1} · seed {spec.seed}",
            spec.prompt[:60],
        ]
        draw.multiline_text(
            (16, 16),
            "\n".join(lines),
            fill="white",
            font=font,
            spacing=6,
            stroke_width=2,
            stroke_fill="black",
        )
        return image


def save_png(image: Image.Image, path: Path) -> None:
    """Save as PNG with no generation metadata (no text chunks, no EXIF; SPEC §12.2)."""
    # Copy pixels into a fresh image so no ``info`` (text, EXIF, ICC) carries over.
    clean = Image.new(image.mode, image.size)
    clean.paste(image)
    clean.save(path, format="PNG")


# ---------------------------------------------------------------------------
# Real Diffusers backend (SPEC §6, §13, §14). torch/diffusers imported lazily.
# ---------------------------------------------------------------------------

# Optimization profiles compared in Phase 8 (SPEC §13.4). "baseline" = FP16 + SDPA.
OPTIMIZATION_PROFILES: tuple[str, ...] = ("baseline", "compile", "compile-max")
DEFAULT_OPTIMIZATION = "baseline"  # until Phase 8 L4 benchmarks pick a winner
WARMUP_STEPS = 3


def configure_cuda_allocator() -> None:
    """Set the allocator config; only effective if called before torch is imported."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def make_scheduler(sampler: str, base_config: Any) -> Any:
    """Build a fresh scheduler for ``sampler`` from the checkpoint's scheduler config."""
    import diffusers

    class_name, overrides = SAMPLER_SCHEDULERS[sampler]
    return getattr(diffusers, class_name).from_config(base_config, **overrides)


class DiffusersBackend:
    """Single-file SD1.5/SDXL checkpoint kept resident on the GPU.

    ``pipeline_loader`` replaces ``from_single_file`` (tests inject tiny random
    pipelines); ``device``/``dtype`` default to the Colab L4 fast path.
    """

    def __init__(
        self,
        family: ModelFamily,
        model_path: Path | str,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        optimization: str = DEFAULT_OPTIMIZATION,
        warmup_steps: int = WARMUP_STEPS,
        warmup_size: tuple[int, int] | None = None,
        log: Callable[[str], None] | None = None,
        pipeline_loader: Callable[[], Any] | None = None,
    ) -> None:
        if optimization not in OPTIMIZATION_PROFILES:
            raise ValueError(f"optimization must be one of {', '.join(OPTIMIZATION_PROFILES)}")
        self.family = check_family(family)
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        self.device = device
        self.dtype_name = dtype
        self.optimization = optimization
        self.active_optimization = "baseline"
        self.warmup_steps = warmup_steps
        defaults = FAMILY_DEFAULTS[self.family]
        self.warmup_size = warmup_size or (defaults.width, defaults.height)
        self.log = log or (lambda message: None)
        self.pipeline_loader = pipeline_loader
        self.batcher = MicroBatcher()
        self.txt2img: Any = None
        self.img2img: Any = None
        self.base_scheduler_config: Any = None
        self._original_modules: dict[str, Any] = {}
        self._seen_shapes: set[BatchKey] = set()

    # -- loading -------------------------------------------------------------

    def load(self, status_callback: StatusCallback | None = None) -> None:
        status = status_callback or (lambda state, message: None)
        configure_cuda_allocator()
        status("loading", "Loading checkpoint…")
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
        else:
            pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
                str(self.model_path), torch_dtype=dtype, add_watermarker=False
            )
        return pipe.to(self.device)

    def _img2img_from(self, pipe: Any) -> Any:
        """Img2img pipeline sharing the already-loaded components (no second weight copy)."""
        import diffusers

        if self.family == "sd15":
            return diffusers.StableDiffusionImg2ImgPipeline.from_pipe(
                pipe, safety_checker=None, requires_safety_checker=False
            )
        img2img = diffusers.StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
        img2img.watermark = None  # SPEC §6.4: never watermark
        return img2img

    # -- optional acceleration -----------------------------------------------

    def _set_module(self, name: str, module: Any) -> None:
        for pipe in (self.txt2img, self.img2img):
            setattr(pipe, name, module)

    def _apply_optimization(self, profile: str) -> None:
        """channels_last + torch.compile; any failure falls back to the baseline."""
        import torch

        self._original_modules = {"unet": self.txt2img.unet, "vae": self.txt2img.vae}
        try:
            mode = "max-autotune" if profile == "compile-max" else "reduce-overhead"
            unet = self.txt2img.unet.to(memory_format=torch.channels_last)
            self._set_module("unet", torch.compile(unet, mode=mode, fullgraph=True))
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
        d = FAMILY_DEFAULTS[self.family]
        width, height = self.warmup_size
        req = GenerationRequest(
            "warm-up", "", width, height, self.warmup_steps, d.guidance_scale,
            d.sampler, 0, 1,
        )  # fmt: skip
        spec = ResolvedImageSpec(0, 0, req.prompt, req.negative_prompt)
        self._run_pipeline(req, [spec], None, lambda step, total: None)
        self._seen_shapes.add((req.mode, req.width, req.height))

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        job: ResolvedGenerationJob,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        if self.txt2img is None:
            raise GenerationError("Backend is not loaded")
        import torch

        req = job.request
        started = time.monotonic()
        now = datetime.now()
        key: BatchKey = (req.mode, req.width, req.height)
        if self.active_optimization != "baseline" and key not in self._seen_shapes:
            self.log(
                f"Optimizing new tensor shape {req.width}x{req.height}; "
                "first generation at this size may be slower."
            )
        self._seen_shapes.add(key)
        source = (
            preprocess_image(req.input_image, req.width, req.height)
            if req.input_image is not None
            else None
        )
        written: list[Path] = []

        def run_batch(
            batch: Sequence[ResolvedImageSpec], on_step: Callable[[int, int], None]
        ) -> list[Path]:
            images = self._run_pipeline(req, batch, source, on_step)
            paths = []
            for spec, image in zip(batch, images, strict=True):
                path = collision_safe_path(
                    output_dir, output_filename(job.job_id, spec.index, spec.seed, now)
                )
                save_png(image, path)
                written.append(path)
                paths.append(path)
            return paths

        def on_oom(failed: int, next_size: int) -> None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log(f"CUDA out of memory at batch {failed}; retrying with batch {next_size}")

        try:
            paths = self.batcher.run(
                key,
                job.images,
                run_batch,
                is_oom=lambda exc: isinstance(exc, torch.OutOfMemoryError),
                progress_callback=progress_callback,
                on_oom=on_oom,
            )
        except Exception as exc:
            raise GenerationError(f"{type(exc).__name__}: {exc}", written) from exc

        return GenerationResult(
            output_paths=tuple(paths),
            seeds=job.seeds,
            resolved_prompts=tuple(spec.prompt for spec in job.images),
            elapsed_seconds=time.monotonic() - started,
        )

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
        embeds = encode_prompt_batch(
            pipe,
            self.family,
            [spec.prompt for spec in batch],
            [spec.negative_prompt for spec in batch],
        )
        # One generator per image: an image depends only on its own seed (SPEC §10).
        generators = [torch.Generator(device=self.device).manual_seed(s.seed) for s in batch]

        def callback(pipeline: Any, step_index: int, timestep: Any, kwargs: dict) -> dict:
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
