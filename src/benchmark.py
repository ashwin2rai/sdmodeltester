"""Phase 8 tooling: optimization-profile benchmark and functional verification.

``python -m src.cli benchmark`` runs this on the Colab L4 and writes a Markdown report
(fields of ``compat/known-good-colab.md``) plus a contact sheet of every functional-check
image for visual review (SPEC §13.4, §23; COLAB_COMPATIBILITY §3). It only uses the public
backend contract, so the mock and tiny CPU pipelines exercise it locally.
"""

from __future__ import annotations

import itertools
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

ALT_SIZES = {"sd15": (768, 512), "sdxl": (896, 1152)}  # "first run at a new resolution"
MIN_SPEEDUP = 0.05  # a compiled profile must beat baseline by this much to be recommended
BLACK_THRESHOLD = 8  # max channel value at or below this => "black image" (fp16 VAE NaNs)


class Runner:
    """Builds requests from family defaults and times ``backend.generate``."""

    def __init__(self, backend: Backend, family: str, output_dir: Path, steps=None, size=None):
        d = FAMILY_DEFAULTS[family]
        self.backend, self.output_dir = backend, output_dir
        self.defaults = dict(
            prompt="a photo of a red fox in a snowy forest, detailed",
            negative_prompt="blurry, lowres",
            width=(size or (d.width, d.height))[0],
            height=(size or (d.width, d.height))[1],
            steps=steps or d.steps,
            guidance_scale=d.guidance_scale,
            sampler=d.sampler,
            seed=1,
            num_images=1,
        )
        self.job_ids = itertools.count(1)
        output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, **overrides: Any) -> tuple[GenerationResult, float]:
        job = resolve_job(GenerationRequest(**{**self.defaults, **overrides}), next(self.job_ids))
        started = time.perf_counter()
        result = self.backend.generate(job, self.output_dir)
        return result, time.perf_counter() - started


def _memory(backend: Backend) -> dict[str, float] | None:
    stats = getattr(backend, "memory_stats", None)
    return stats() if stats else None


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
    runner = Runner(backend, family, output_dir, steps, size)
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = fn()
        timings[name] = round(time.perf_counter() - started, 3)
        log(f"benchmark {profile}: {name} {timings[name]:.2f}s")
        return value

    timed("load", backend.load)
    timed("warmup", backend.warmup)
    first = timed("warm_single_1", lambda: runner.run(seed=1)[0]).output_paths[0]
    timed("warm_single_2", lambda: runner.run(seed=2))
    timed("warm_single_3", lambda: runner.run(seed=3))
    if reset := getattr(backend, "reset_peak_memory", None):
        reset()
    for n in (5, 10):
        timed(f"batch_{n}", lambda n=n: runner.run(seed=100, num_images=n))
    peak = _memory(backend)
    alt_w, alt_h = alt_size or ALT_SIZES[family]
    for which in ("first", "second"):
        timed(f"new_size_{which}", lambda: runner.run(width=alt_w, height=alt_h))
    strength = FAMILY_DEFAULTS[family].strength
    for which in ("first", "second"):
        timed(f"img2img_{which}", lambda: runner.run(input_image=first, strength=strength))

    return {
        "profile_active": getattr(backend, "active_optimization", profile),
        "timings_s": timings,
        "warm_single_median_s": statistics.median(
            [timings["warm_single_2"], timings["warm_single_3"]]
        ),
        "startup_s": timings["load"] + timings["warmup"],
        "peak_vram_batches": peak,
        "batch_limits": {f"{m} {w}x{h}": n for (m, w, h), n in backend.batcher.limits.items()},
        "settings": f"{runner.defaults['steps']} steps, "
        f"{runner.defaults['width']}x{runner.defaults['height']} (new size {alt_w}x{alt_h})",
    }


def recommend(results: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """Fastest warm single-image latency among profiles that ran and stayed active; a
    compiled profile must beat baseline by ``MIN_SPEEDUP`` to be worth its startup cost."""
    ok = {n: r for n, r in results.items() if "error" not in r and r["profile_active"] == n}
    if not ok:
        return None, "no profile completed"
    best = min(ok, key=lambda n: ok[n]["warm_single_median_s"])
    base = ok.get("baseline")
    if not base or best == "baseline":
        return best, f"{best} has the lowest warm single-image latency"
    gain = 1 - ok[best]["warm_single_median_s"] / base["warm_single_median_s"]
    if gain < MIN_SPEEDUP:
        return "baseline", f"{best} is only {gain:.0%} faster than baseline; keeping baseline"
    return best, (
        f"{best} is {gain:.0%} faster per warm image than baseline; "
        f"startup {ok[best]['startup_s']:.0f}s vs {base['startup_s']:.0f}s"
    )


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
    """Run the functional matrix on a *loaded* backend; failures are recorded, not raised."""
    r = Runner(backend, family, output_dir, steps, size)
    d = FAMILY_DEFAULTS[family]
    alt_w, alt_h = alt_size or ALT_SIZES[family]
    first: list[Path] = []

    def sized(result: GenerationResult, w: int, h: int) -> bool:
        for path in result.output_paths:
            with Image.open(path) as img:
                if img.size != (w, h):
                    return False
        return True

    def txt2img(n: int):
        result, secs = r.run(seed=500, num_images=n)
        first.append(result.output_paths[0])
        ok = result.seeds == tuple(range(500, 500 + n))
        ok = ok and sized(result, r.defaults["width"], r.defaults["height"])
        return ok, f"{n} image(s) in {secs:.1f}s", result.output_paths

    def img2img(n: int):
        result, secs = r.run(input_image=first[0], strength=d.strength, seed=600, num_images=n)
        ok = len(result.output_paths) == n
        return ok and sized(result, r.defaults["width"], r.defaults["height"]), \
            f"{n} image(s) in {secs:.1f}s", result.output_paths  # fmt: skip

    def sampler(sid: str):
        result, secs = r.run(sampler=sid, seed=700)
        return len(result.output_paths) == 1, f"{secs:.1f}s", result.output_paths

    def random_seeds():
        result, _ = r.run(seed=-1, num_images=3)
        return -1 not in result.seeds and len(set(result.seeds)) == 3, \
            f"seeds {list(result.seeds)}", result.output_paths  # fmt: skip

    def reproducible():
        a, b = r.run(seed=42)[0], r.run(seed=42)[0]
        diff = max_pixel_diff(a.output_paths[0], b.output_paths[0])
        return diff <= 2, f"same seed twice: max pixel diff {diff}", a.output_paths

    def batch_invariance():  # informational tolerance: GPU batched kernels may differ slightly
        batch, alone = r.run(seed=42, num_images=2)[0], r.run(seed=43)[0]
        diff = max_pixel_diff(batch.output_paths[1], alone.output_paths[0])
        return diff <= 24, f"seed 43 in batch vs alone: max pixel diff {diff}", ()

    def weighted():
        plain = r.run(prompt="a red fox, snow", seed=800)[0].output_paths[0]
        heavy = r.run(prompt="a (red:1.6) fox, (snow:0.6)", seed=800)[0].output_paths[0]
        diff = max_pixel_diff(plain, heavy)
        return diff > 0, f"weighted vs plain differ (max diff {diff})", (plain, heavy)

    def dynamic(prompt: str, allowed: set[str], n: int):
        result, _ = r.run(prompt=prompt, seed=900, num_images=n)
        return set(result.resolved_prompts) <= allowed, \
            f"resolved {list(result.resolved_prompts)}", result.output_paths  # fmt: skip

    def non_square():
        result, secs = r.run(width=alt_w, height=alt_h, seed=950)
        return sized(result, alt_w, alt_h), f"{alt_w}x{alt_h} in {secs:.1f}s", result.output_paths

    def no_reload():
        before = backend.load_count
        times = [r.run(seed=960 + i)[1] for i in range(3)]
        return (
            backend.load_count == before,
            "latencies " + ", ".join(f"{t:.2f}s" for t in times),
            (),
        )

    colors = ("red", "white", "black")
    matrix = [
        ("txt2img x1", lambda: txt2img(1)),
        ("txt2img x10", lambda: txt2img(10)),
        ("img2img x1", lambda: img2img(1)),
        ("img2img x10", lambda: img2img(10)),
        *[(f"sampler {SAMPLER_LABELS[s]}", lambda s=s: sampler(s)) for s in SAMPLER_IDS],
        ("seed -1 random per image", random_seeds),
        ("fixed seed reproducible", reproducible),
        ("batch invariance", batch_invariance),
        ("weighted prompt", weighted),
        ("dynamic prompt", lambda: dynamic("a {red | white | black} fox",
                                           {f"a {c} fox" for c in colors}, 4)),
        ("dynamic + weighted", lambda: dynamic("a {(red:1.4) | white} fox",
                                               {"a (red:1.4) fox", "a white fox"}, 2)),
        ("non-square", non_square),
        ("repeated singles, no reload", no_reload),
    ]  # fmt: skip

    checks = []
    for name, fn in matrix:
        try:
            ok, detail, images = fn()
        except Exception as exc:  # noqa: BLE001 — record and continue
            ok, detail, images = False, f"{type(exc).__name__}: {exc}", ()
        if black := [p.name for p in images if is_black(p)]:
            ok, detail = False, f"black image(s): {', '.join(black)}; {detail}"
        checks.append(Check(name, ok, detail, list(images)))
        log(f"verify: {name}: {'ok' if ok else 'FAIL'} {detail}")
    return checks


def contact_sheet(checks: list[Check] | None, path: Path, thumb: int = 160) -> Path | None:
    """One labelled row per check (up to 10 thumbnails) for human review of image quality."""
    rows = [c for c in checks or () if c.images]
    if not rows:
        return None
    label_w, step = 220, thumb + 4
    sheet = Image.new("RGB", (label_w + 10 * step, len(rows) * step), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for row, c in enumerate(rows):
        draw.multiline_text((6, row * step + 6), f"{c.name}\n{'ok' if c.ok else 'FAIL'}",
                            fill="black" if c.ok else "red", font=font)  # fmt: skip
        for i, image_path in enumerate(c.images[:10]):
            with Image.open(image_path) as img:
                img.thumbnail((thumb, thumb))
                sheet.paste(img.convert("RGB"), (label_w + i * step, row * step))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

ROWS = [
    ("active profile", lambda r: r["profile_active"]),
    ("cold load (s)", lambda r: r["timings_s"]["load"]),
    ("compile/warm-up (s)", lambda r: r["timings_s"]["warmup"]),
    *[(f"warm #{i} (s)", lambda r, i=i: r["timings_s"][f"warm_single_{i}"]) for i in (1, 2, 3)],
    ("batch 5 total (s)", lambda r: r["timings_s"]["batch_5"]),
    ("batch 10 total (s)", lambda r: r["timings_s"]["batch_10"]),
    ("batch 10 images/s", lambda r: round(10 / r["timings_s"]["batch_10"], 2)),
    ("new size first/second (s)",
     lambda r: f"{r['timings_s']['new_size_first']}/{r['timings_s']['new_size_second']}"),
    ("img2img first/second (s)",
     lambda r: f"{r['timings_s']['img2img_first']}/{r['timings_s']['img2img_second']}"),
    ("peak VRAM in batches (GiB alloc/reserved)",
     lambda r: "{peak_allocated_gib}/{peak_reserved_gib}".format(**r["peak_vram_batches"])
     if r["peak_vram_batches"] else "—"),
    ("OOM batch limits", lambda r: r["batch_limits"] or "none"),
]  # fmt: skip


def render_report(
    *,
    family: str,
    model_name: str,
    environment: list[tuple[str, str]],
    profiles: dict[str, dict[str, Any]],
    checks: list[Check] | None,
    sheet: Path | None,
) -> str:
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
        lines += [
            "## Performance",
            "",
            "| | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for label, get in ROWS:
            cells = ["error" if "error" in profiles[n] else str(get(profiles[n])) for n in names]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += [f"\n- **{n} failed:** {r['error']}" for n, r in profiles.items() if "error" in r]
        if settings := next((r["settings"] for r in profiles.values() if "settings" in r), None):
            lines.append(f"\nSettings: {settings}.")
        choice, reason = recommend(profiles)
        lines += ["", f"**Recommended default profile:** {choice or '—'} — {reason}", ""]
    if checks is not None:
        lines += ["## Functional checks", "", f"{sum(c.ok for c in checks)}/{len(checks)} passed.",
                  "", "| check | result | detail |", "|---|---|---|"]  # fmt: skip
        for c in checks:
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {'ok' if c.ok else '**FAIL**'} | {detail} |")
        if sheet:
            lines += ["", f"Contact sheet for visual review: `{sheet}`"]
        lines.append("")
    return "\n".join(lines)


def default_report_name(family: str) -> str:
    return f"benchmark-{family}-{date.today().isoformat()}.md"
