"""Phase 8 tooling: optimization-profile benchmark and functional verification.

Runs on the Colab L4 (``python -m src.cli benchmark ...``) and writes a Markdown report
shaped like ``compat/known-good-colab.md`` plus a JSON dump and a contact sheet of every
functional-check image for visual review (SPEC §13.4, §23; COLAB_COMPATIBILITY §3).

Everything here drives the public backend contract (load / warmup / generate), so it is
exercised locally with the mock backend and the tiny CPU Diffusers pipelines.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.backend import (
    FAMILY_DEFAULTS,
    SAMPLER_IDS,
    SAMPLER_LABELS,
    Backend,
    GenerationRequest,
    GenerationResult,
    resolve_job,
)

# Non-default shapes for the "first generation at a new resolution" measurement.
ALT_SIZES = {"sd15": (768, 512), "sdxl": (896, 1152)}
# Accept profile B/C over baseline only if warm latency improves by at least this much.
MIN_SPEEDUP = 0.05
BLACK_THRESHOLD = 8  # max channel value at or below this => "black image" failure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    backend: Backend
    family: str
    output_dir: Path
    steps: int | None = None  # override the family default (tests use tiny values)
    size: tuple[int, int] | None = None  # override the family default size
    _job_ids: Any = field(default_factory=lambda: iter(range(1, 10**9)))

    def request(self, **overrides: Any) -> GenerationRequest:
        d = FAMILY_DEFAULTS[self.family]
        width, height = self.size or (d.width, d.height)
        fields = dict(
            prompt="a photo of a red fox in a snowy forest, detailed",
            negative_prompt="blurry, lowres",
            width=width,
            height=height,
            steps=self.steps or d.steps,
            guidance_scale=d.guidance_scale,
            sampler=d.sampler,
            seed=1,
            num_images=1,
        )
        fields.update(overrides)
        return GenerationRequest(**fields)

    def run(self, req: GenerationRequest) -> tuple[GenerationResult, float]:
        job = resolve_job(req, job_id=next(self._job_ids))
        started = time.perf_counter()
        result = self.backend.generate(job, self.output_dir)
        return result, time.perf_counter() - started


def _memory(backend: Backend) -> dict[str, float] | None:
    stats = getattr(backend, "memory_stats", None)
    return stats() if stats else None


def _reset_memory(backend: Backend) -> None:
    reset = getattr(backend, "reset_peak_memory", None)
    if reset:
        reset()


def is_black(path: Path) -> bool:
    with Image.open(path) as img:
        return max(high for _, high in img.convert("RGB").getextrema()) <= BLACK_THRESHOLD


def max_pixel_diff(a: Path, b: Path) -> int:
    with Image.open(a) as x, Image.open(b) as y:
        if x.size != y.size:
            return 255
        return max(abs(p - q) for p, q in zip(x.tobytes(), y.tobytes(), strict=True))


# ---------------------------------------------------------------------------
# Performance (SPEC §23.2)
# ---------------------------------------------------------------------------


def benchmark_profile(
    backend: Backend,
    family: str,
    output_dir: Path,
    *,
    profile: str = "baseline",
    steps: int | None = None,
    size: tuple[int, int] | None = None,
    alt_size: tuple[int, int] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure one optimization profile on an *unloaded* backend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(backend, family, output_dir, steps=steps, size=size)
    timings: dict[str, Any] = {}
    memory: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], Any]) -> Any:
        log(f"benchmark: {name}…")
        started = time.perf_counter()
        value = fn()
        timings[name] = round(time.perf_counter() - started, 3)
        log(f"benchmark: {name} {timings[name]:.2f}s")
        return value

    _reset_memory(backend)
    step("load", backend.load)
    memory["after_load"] = _memory(backend)
    step("warmup", backend.warmup)
    active = getattr(backend, "active_optimization", profile)

    first_path = None
    for i in (1, 2, 3):
        result = step(f"warm_single_{i}", lambda i=i: ctx.run(ctx.request(seed=i))[0])
        first_path = first_path or result.output_paths[0]
    memory["single"] = _memory(backend)

    for n in (5, 10):
        _reset_memory(backend)
        step(f"batch_{n}", lambda n=n: ctx.run(ctx.request(seed=100, num_images=n)))
        timings[f"batch_{n}_images_per_s"] = round(n / timings[f"batch_{n}"], 3)
        memory[f"batch_{n}"] = _memory(backend)

    alt_w, alt_h = alt_size or ALT_SIZES[family]
    step("new_size_first", lambda: ctx.run(ctx.request(width=alt_w, height=alt_h)))
    step("new_size_second", lambda: ctx.run(ctx.request(width=alt_w, height=alt_h, seed=2)))
    d = FAMILY_DEFAULTS[family]
    step(
        "img2img_first",
        lambda: ctx.run(ctx.request(input_image=first_path, strength=d.strength)),
    )
    step(
        "img2img_second",
        lambda: ctx.run(ctx.request(input_image=first_path, strength=d.strength, seed=2)),
    )

    warm = [timings["warm_single_2"], timings["warm_single_3"]]
    return {
        "profile_requested": profile,
        "profile_active": active,
        "timings_s": timings,
        "warm_single_median_s": round(statistics.median(warm), 3),
        "startup_s": round(timings["load"] + timings["warmup"], 3),
        "memory": memory,
        "batch_limits": {
            f"{mode} {w}x{h}": size for (mode, w, h), size in backend.batcher.limits.items()
        },
        "steps": ctx.steps or d.steps,
        "size": list(ctx.size or (d.width, d.height)),
        "alt_size": [alt_w, alt_h],
    }


def recommend(results: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """Pick the default profile: fastest warm single-image latency that is stable.

    A compiled profile must beat baseline by ``MIN_SPEEDUP`` to be worth its startup
    cost; profiles that errored or fell back to baseline are not candidates.
    """
    ok = {
        name: r
        for name, r in results.items()
        if "error" not in r and r.get("profile_active") == name
    }
    if not ok:
        return None, "no profile completed"
    best = min(ok, key=lambda name: ok[name]["warm_single_median_s"])
    baseline = ok.get("baseline")
    if baseline and best != "baseline":
        gain = 1 - ok[best]["warm_single_median_s"] / baseline["warm_single_median_s"]
        if gain < MIN_SPEEDUP:
            return "baseline", (
                f"{best} is only {gain:.0%} faster than baseline (< {MIN_SPEEDUP:.0%}); "
                "keeping baseline"
            )
        return best, (
            f"{best} is {gain:.0%} faster per warm image than baseline; startup "
            f"{ok[best]['startup_s']:.0f}s vs {baseline['startup_s']:.0f}s"
        )
    return best, f"{best} has the lowest warm single-image latency"


# ---------------------------------------------------------------------------
# Functional matrix (SPEC §23.1)
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    images: list[Path] = field(default_factory=list)


def functional_checks(
    backend: Backend,
    family: str,
    output_dir: Path,
    *,
    steps: int | None = None,
    size: tuple[int, int] | None = None,
    alt_size: tuple[int, int] | None = None,
    log: Callable[[str], None] = print,
) -> list[Check]:
    """Run the functional matrix on a *loaded* backend; never raises per check."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(backend, family, output_dir, steps=steps, size=size)
    d = FAMILY_DEFAULTS[family]
    width, height = ctx.size or (d.width, d.height)
    checks: list[Check] = []
    state: dict[str, Path] = {}

    def check(name: str, fn: Callable[[], tuple[bool, str, list[Path]]]) -> None:
        log(f"verify: {name}…")
        try:
            ok, detail, images = fn()
        except Exception as exc:  # noqa: BLE001 — record and continue
            ok, detail, images = False, f"{type(exc).__name__}: {exc}", []
        black = [p.name for p in images if is_black(p)]
        if black:
            ok, detail = False, f"black image(s): {', '.join(black)}; {detail}"
        checks.append(Check(name, ok, detail, images))
        log(f"verify: {name}: {'ok' if ok else 'FAIL'} {detail}")

    def sizes_ok(result: GenerationResult, w: int, h: int) -> bool:
        for path in result.output_paths:
            with Image.open(path) as img:
                if img.size != (w, h):
                    return False
        return True

    def txt2img(n: int) -> tuple[bool, str, list[Path]]:
        result, secs = ctx.run(ctx.request(seed=500, num_images=n))
        state.setdefault("source", result.output_paths[0])
        expected = tuple(range(500, 500 + n))
        ok = len(result.output_paths) == n and result.seeds == expected
        ok = ok and sizes_ok(result, width, height)
        return ok, f"{n} image(s) in {secs:.1f}s", list(result.output_paths)

    def img2img(n: int) -> tuple[bool, str, list[Path]]:
        result, secs = ctx.run(
            ctx.request(input_image=state["source"], strength=d.strength, seed=600, num_images=n)
        )
        ok = len(result.output_paths) == n and sizes_ok(result, width, height)
        return ok, f"{n} image(s) in {secs:.1f}s", list(result.output_paths)

    check("txt2img x1", lambda: txt2img(1))
    check("txt2img x10", lambda: txt2img(10))
    check("img2img x1", lambda: img2img(1))
    check("img2img x10", lambda: img2img(10))

    for sampler in SAMPLER_IDS:

        def run_sampler(sampler: str = sampler) -> tuple[bool, str, list[Path]]:
            result, secs = ctx.run(ctx.request(sampler=sampler, seed=700))
            return len(result.output_paths) == 1, f"{secs:.1f}s", list(result.output_paths)

        check(f"sampler {SAMPLER_LABELS[sampler]}", run_sampler)

    def random_seeds() -> tuple[bool, str, list[Path]]:
        result, _ = ctx.run(ctx.request(seed=-1, num_images=3))
        ok = -1 not in result.seeds and len(set(result.seeds)) == 3
        return ok, f"seeds {list(result.seeds)}", list(result.output_paths)

    def reproducible() -> tuple[bool, str, list[Path]]:
        a, _ = ctx.run(ctx.request(seed=42))
        b, _ = ctx.run(ctx.request(seed=42))
        diff = max_pixel_diff(a.output_paths[0], b.output_paths[0])
        return diff <= 2, f"same seed twice: max pixel diff {diff}", [a.output_paths[0]]

    def batch_invariance() -> tuple[bool, str, list[Path]]:
        batch, _ = ctx.run(ctx.request(seed=42, num_images=2))
        alone, _ = ctx.run(ctx.request(seed=43))
        diff = max_pixel_diff(batch.output_paths[1], alone.output_paths[0])
        # informational: GPU batched kernels may differ slightly; large diffs are a bug
        return diff <= 24, f"seed 43 in batch vs alone: max pixel diff {diff}", []

    def weighted() -> tuple[bool, str, list[Path]]:
        plain, _ = ctx.run(ctx.request(prompt="a red fox, snow", seed=800))
        heavy, _ = ctx.run(ctx.request(prompt="a (red:1.6) fox, (snow:0.6)", seed=800))
        diff = max_pixel_diff(plain.output_paths[0], heavy.output_paths[0])
        return (
            diff > 0,
            f"weighted vs plain differ (max diff {diff})",
            [
                plain.output_paths[0],
                heavy.output_paths[0],
            ],
        )

    def dynamic() -> tuple[bool, str, list[Path]]:
        result, _ = ctx.run(
            ctx.request(prompt="a {red | white | black} fox", seed=900, num_images=4)
        )
        allowed = {f"a {c} fox" for c in ("red", "white", "black")}
        ok = set(result.resolved_prompts) <= allowed
        return ok, f"resolved {list(result.resolved_prompts)}", list(result.output_paths)

    def combined() -> tuple[bool, str, list[Path]]:
        result, _ = ctx.run(ctx.request(prompt="a {(red:1.4) | white} fox", seed=910, num_images=2))
        ok = all("{" not in p for p in result.resolved_prompts)
        return ok, f"resolved {list(result.resolved_prompts)}", list(result.output_paths)

    def non_square() -> tuple[bool, str, list[Path]]:
        w, h = alt_size or ALT_SIZES[family]
        result, secs = ctx.run(ctx.request(width=w, height=h, seed=950))
        return sizes_ok(result, w, h), f"{w}x{h} in {secs:.1f}s", list(result.output_paths)

    def repeated() -> tuple[bool, str, list[Path]]:
        loads = getattr(backend, "load_count", None)
        times = [ctx.run(ctx.request(seed=960 + i))[1] for i in range(3)]
        detail = "latencies " + ", ".join(f"{t:.2f}s" for t in times)
        after = getattr(backend, "load_count", None)
        return loads == after, detail, []

    check("seed -1 random per image", random_seeds)
    check("fixed seed reproducible", reproducible)
    check("batch invariance", batch_invariance)
    check("weighted prompt", weighted)
    check("dynamic prompt", dynamic)
    check("dynamic + weighted", combined)
    check("non-square", non_square)
    check("repeated singles, no reload", repeated)
    return checks


def contact_sheet(checks: list[Check], path: Path, thumb: int = 160) -> Path | None:
    """One labelled row per check, up to 10 thumbnails each, for visual review."""
    rows = [(c, c.images[:10]) for c in checks if c.images]
    if not rows:
        return None
    label_w = 220
    sheet = Image.new("RGB", (label_w + 10 * (thumb + 4), len(rows) * (thumb + 4)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for r, (c, images) in enumerate(rows):
        y = r * (thumb + 4)
        color = "black" if c.ok else "red"
        draw.multiline_text(
            (6, y + 6), f"{c.name}\n{'ok' if c.ok else 'FAIL'}", fill=color, font=font
        )
        for i, image_path in enumerate(images):
            with Image.open(image_path) as img:
                img.thumbnail((thumb, thumb))
                sheet.paste(img.convert("RGB"), (label_w + i * (thumb + 4), y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_report(
    *,
    family: str,
    model_name: str,
    environment: list[tuple[str, str]],
    profiles: dict[str, dict[str, Any]],
    checks: list[Check] | None,
    sheet: Path | None,
) -> str:
    choice, reason = recommend(profiles) if profiles else (None, "not benchmarked")
    lines = [
        f"# Benchmark: {family} · {model_name}",
        "",
        f"Date tested: {date.today().isoformat()}",
        "",
        "## Environment",
        "",
        *[f"- {name}: {value}" for name, value in environment],
        "",
    ]
    if profiles:
        names = list(profiles)
        rows = [
            ("active profile", lambda r: r.get("profile_active")),
            ("cold load (s)", lambda r: r["timings_s"].get("load")),
            ("compile/warm-up (s)", lambda r: r["timings_s"].get("warmup")),
            ("warm #1 (s)", lambda r: r["timings_s"].get("warm_single_1")),
            ("warm #2 (s)", lambda r: r["timings_s"].get("warm_single_2")),
            ("warm #3 (s)", lambda r: r["timings_s"].get("warm_single_3")),
            ("batch 5 total (s)", lambda r: r["timings_s"].get("batch_5")),
            ("batch 10 total (s)", lambda r: r["timings_s"].get("batch_10")),
            ("batch 10 images/s", lambda r: r["timings_s"].get("batch_10_images_per_s")),
            ("new size first (s)", lambda r: r["timings_s"].get("new_size_first")),
            ("new size second (s)", lambda r: r["timings_s"].get("new_size_second")),
            ("img2img first (s)", lambda r: r["timings_s"].get("img2img_first")),
            ("img2img second (s)", lambda r: r["timings_s"].get("img2img_second")),
            (
                "peak VRAM batch 10 alloc/reserved (GiB)",
                lambda r: (
                    "{peak_allocated_gib}/{peak_reserved_gib}".format(**r["memory"]["batch_10"])
                    if (r.get("memory") or {}).get("batch_10")
                    else None
                ),
            ),
            ("OOM batch limits", lambda r: r.get("batch_limits") or "none"),
        ]
        lines += [
            "## Performance",
            "",
            "| | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for label, get in rows:
            cells = []
            for name in names:
                result = profiles[name]
                cells.append("error" if "error" in result else _fmt(get(result)))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        errors = {n: r["error"] for n, r in profiles.items() if "error" in r}
        for name, error in errors.items():
            lines.append(f"\n- **{name} failed:** {error}")
        first = next(iter(profiles.values()))
        if "steps" in first:
            lines.append(
                f"\nSettings: {first['steps']} steps, {first['size'][0]}x{first['size'][1]} "
                f"(new size {first['alt_size'][0]}x{first['alt_size'][1]})."
            )
        lines += ["", f"**Recommended default profile:** {choice or '—'} — {reason}", ""]
    if checks is not None:
        passed = sum(c.ok for c in checks)
        lines += [
            "## Functional checks",
            "",
            f"{passed}/{len(checks)} passed.",
            "",
            "| check | result | detail |",
            "|---|---|---|",
        ]
        for c in checks:
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {'ok' if c.ok else '**FAIL**'} | {detail} |")
        if sheet:
            lines += ["", f"Contact sheet for visual review: `{sheet}`"]
        lines.append("")
    return "\n".join(lines)


def environment_rows(diagnostics: list[Any]) -> list[tuple[str, str]]:
    rows = [(c.name, c.value) for c in diagnostics]
    if not any(name == "Python" for name, _ in rows):
        rows.insert(0, ("Python", platform.python_version()))
    return rows


def default_report_name(family: str) -> str:
    return f"benchmark-{family}-{date.today().isoformat()}.md"
