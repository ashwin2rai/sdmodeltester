"""Command-line interface: ``doctor``, ``generate``, ``serve``.

    python -m src.cli doctor [--mock] [--model PATH --model-family sd15|sdxl] [--compile-check]
    python -m src.cli generate --model-family sdxl --model models/x.safetensors --prompt "..."
    python -m src.cli generate --mock --model-family sd15 --prompt "a {white | black} cat"
    python -m src.cli serve --model-family sdxl --model models/x.safetensors --port 8000
    python -m src.cli serve --mock --model-family sdxl --port 8000

Mock mode never imports torch. Real mode sets the CUDA allocator config before torch is
first imported (SPEC §13.2).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.backend import (
    DEFAULT_OPTIMIZATION,
    FAMILY_DEFAULTS,
    MODEL_FAMILIES,
    OPTIMIZATION_PROFILES,
    SAMPLER_IDS,
    SAMPLER_LABELS,
    Backend,
    DiffusersBackend,
    GenerationError,
    GenerationRequest,
    MockBackend,
    ValidationError,
    configure_cuda_allocator,
    resolve_job,
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

Printer = Callable[[str], None]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_model_args(parser: argparse.ArgumentParser, *, family_required: bool) -> None:
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        required=family_required,
        help="explicit model family (no auto-detection)",
    )
    parser.add_argument("--model", type=Path, help="path to a single-file .safetensors checkpoint")


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mock", action="store_true", help="placeholder backend: no GPU/torch")
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", default="float16", help=argparse.SUPPRESS)
    parser.add_argument(
        "--optimization",
        choices=OPTIMIZATION_PROFILES,
        default=DEFAULT_OPTIMIZATION,
        help=f"acceleration profile (default: {DEFAULT_OPTIMIZATION})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="report runtime/GPU capabilities")
    _add_model_args(doctor, family_required=False)
    doctor.add_argument("--mock", action="store_true", help="check only mock-mode requirements")
    doctor.add_argument(
        "--compile-check", action="store_true", help="also run a small torch.compile smoke test"
    )

    gen = sub.add_parser("generate", help="generate images once and exit")
    _add_model_args(gen, family_required=True)
    _add_backend_args(gen)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--width", type=int, help="default: family default")
    gen.add_argument("--height", type=int, help="default: family default")
    gen.add_argument("--steps", type=int, help="default: family default")
    gen.add_argument("--cfg", type=float, help="guidance scale (default: family default)")
    gen.add_argument("--sampler", choices=SAMPLER_IDS, help="default: family default")
    gen.add_argument("--seed", type=int, default=-1, help="-1 = random per image (default)")
    gen.add_argument("--images", type=int, default=1, help="1-10 (default: 1)")
    gen.add_argument("--image", type=Path, help="input image for image-to-image")
    gen.add_argument("--strength", type=float, help="img2img strength (default: family default)")
    gen.add_argument("--output-dir", type=Path, default=Path("outputs"))

    bench = sub.add_parser(
        "benchmark", help="Phase 8: compare optimization profiles + functional checks"
    )
    _add_model_args(bench, family_required=True)
    bench.add_argument("--mock", action="store_true", help="placeholder backend: no GPU/torch")
    bench.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    bench.add_argument("--dtype", default="float16", help=argparse.SUPPRESS)
    bench.add_argument(
        "--profiles",
        default="baseline,compile",
        help="comma-separated optimization profiles to benchmark "
        f"({', '.join(OPTIMIZATION_PROFILES)}; default: baseline,compile)",
    )
    bench.add_argument(
        "--verify-profile",
        help="profile whose process also runs the functional checks (default: first)",
    )
    bench.add_argument("--no-functional", action="store_true", help="skip functional checks")
    bench.add_argument("--no-perf", action="store_true", help="only run functional checks")
    bench.add_argument("--steps", type=int, help="override steps (default: family default)")
    bench.add_argument("--width", type=int, help="override width (default: family default)")
    bench.add_argument("--height", type=int, help="override height (default: family default)")
    bench.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    bench.add_argument("--report", type=Path, help="Markdown report path (default: compat/…)")
    bench.add_argument("--in-process", action="store_true", help="run all profiles in this process")
    bench.add_argument(
        "--quiet",
        action="store_true",
        help="no progress output; print only the report path (the JSON next to it lists "
        "the contact sheet)",
    )
    bench.add_argument("--child-json", type=Path, help=argparse.SUPPRESS)

    serve = sub.add_parser("serve", help="run the HTTP server and browser UI")
    _add_model_args(serve, family_required=True)
    _add_backend_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--inputs-dir", type=Path, default=Path("inputs"))
    serve.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    serve.add_argument(
        "--mock-load-seconds", type=float, default=2.0, help="mock: simulated load/warm-up delay"
    )
    serve.add_argument(
        "--mock-step-seconds", type=float, default=0.05, help="mock: simulated time per step"
    )
    return parser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class UsageError(Exception):
    """Bad combination of arguments (exit code 2)."""


def check_model_path(model: Path | None) -> Path:
    if model is None:
        raise UsageError("--model is required unless --mock is given")
    if model.suffix.lower() != ".safetensors":
        raise UsageError(f"--model must be a .safetensors file (got {model.name})")
    if not model.is_file():
        raise UsageError(f"checkpoint not found: {model}")
    return model


def build_backend(
    args: argparse.Namespace, log: Printer, *, mock_delays: tuple[float, float] = (0.0, 0.0)
) -> Backend:
    if args.mock:
        load_seconds, step_seconds = mock_delays
        name = args.model.name if args.model else "mock.safetensors"
        return MockBackend(
            args.model_family, name, load_seconds=load_seconds, step_seconds=step_seconds
        )
    model = check_model_path(args.model)
    configure_cuda_allocator()  # must precede the first torch import
    return DiffusersBackend(
        args.model_family,
        model,
        device=args.device,
        dtype=args.dtype,
        optimization=args.optimization,
        log=log,
    )


def build_request(args: argparse.Namespace) -> tuple[GenerationRequest, list[str]]:
    """Fill unset knobs from family defaults; returns the request and CLI notes."""
    d = FAMILY_DEFAULTS[args.model_family]
    notes = []
    strength = None
    if args.image is not None:
        if not args.image.is_file():
            raise UsageError(f"input image not found: {args.image}")
        strength = d.strength if args.strength is None else args.strength
    elif args.strength is not None:
        notes.append("--strength ignored: no --image given (text-to-image)")

    def pick(value: Any, default: Any) -> Any:
        return default if value is None else value

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=pick(args.width, d.width),
        height=pick(args.height, d.height),
        steps=pick(args.steps, d.steps),
        guidance_scale=pick(args.cfg, d.guidance_scale),
        sampler=pick(args.sampler, d.sampler),
        seed=args.seed,
        num_images=args.images,
        input_image=args.image,
        strength=strength,
    )
    return request, notes


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    request, notes = build_request(args)
    job = resolve_job(request, job_id=1)
    req = job.request
    for note in (*notes, *job.corrections):
        err(f"Note: {note}")

    backend = build_backend(args, err)
    try:
        backend.load(lambda state, message: err(f"Backend: {message}"))
        if not args.mock and getattr(backend, "optimization", "baseline") != "baseline":
            backend.warmup(lambda state, message: err(f"Backend: {message}"))
    except Exception as exc:  # noqa: BLE001 — report load failures concisely
        err(f"Error: backend failed to load: {type(exc).__name__}: {exc}")
        return EXIT_FAILURE

    mode = "img2img" if req.input_image else "txt2img"
    detail = f" · strength {req.strength}" if req.input_image else ""
    err(
        f"Job 1: {mode} · {SAMPLER_LABELS[req.sampler]} · {req.steps} steps · "
        f"{req.width}x{req.height} · CFG {req.guidance_scale}{detail} · "
        f"{req.num_images} image(s) · seeds {', '.join(map(str, job.seeds))}"
    )
    if any(spec.prompt != req.prompt for spec in job.images):
        for spec in job.images:
            err(f"  image {spec.index + 1} (seed {spec.seed}): {spec.prompt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_decile = [-1]

    def progress(fraction: float, message: str) -> None:
        decile = int(fraction * 10)  # report roughly every 10%
        if decile != last_decile[0]:
            last_decile[0] = decile
            err(f"  {message} ({int(fraction * 100)}%)")

    try:
        result = backend.generate(job, args.output_dir, progress)
    except GenerationError as exc:
        err(f"Error: {exc}")
        for path in exc.completed_paths:
            err(f"  completed before failure: {path}")
        return EXIT_FAILURE

    for path in result.output_paths:
        out(str(path))
    err(f"Done in {result.elapsed_seconds:.1f}s")
    return EXIT_OK


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    backend = build_backend(args, err, mock_delays=(args.mock_load_seconds, args.mock_step_seconds))
    try:
        from src import server
    except ImportError:
        err("Error: the HTTP server is not implemented yet (Phase 5).")
        return EXIT_FAILURE
    server.serve(
        backend,
        host=args.host,
        port=args.port,
        inputs_dir=args.inputs_dir,
        outputs_dir=args.outputs_dir,
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _bench_size(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.width is None and args.height is None:
        return None
    d = FAMILY_DEFAULTS[args.model_family]
    return (args.width or d.width, args.height or d.height)


def _run_profile(args: argparse.Namespace, profile: str, verify: bool, err: Printer) -> dict:
    """Benchmark one profile (and optionally the functional matrix) in this process."""
    from src import benchmark

    profile_args = argparse.Namespace(**{**vars(args), "optimization": profile})
    size = _bench_size(args)
    alt = None if size is None else (size[0] + 64, size[1])  # a different, non-square shape
    out: dict[str, Any] = {"profile": profile}
    out["environment"] = [
        dataclasses.astuple(c)[:2]
        for c in collect_diagnostics(mock=args.mock, model=args.model, family=args.model_family)
    ]
    backend = build_backend(profile_args, err)
    common = dict(steps=args.steps, size=size, alt_size=alt, log=err)
    try:
        if not args.no_perf:
            out["perf"] = benchmark.benchmark_profile(
                backend,
                args.model_family,
                args.output_dir / profile,
                profile=profile,
                **common,
            )
        if verify:
            if args.no_perf:
                backend.load()
                backend.warmup()
            checks = benchmark.functional_checks(
                backend, args.model_family, args.output_dir / "functional", **common
            )
            out["checks"] = [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "images": [str(p) for p in c.images],
                }
                for c in checks
            ]
    except Exception as exc:  # noqa: BLE001 — a crashed profile is a result, not a crash
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _run_profile_subprocess(
    args: argparse.Namespace, argv: Sequence[str], profile: str, verify: bool, err: Printer
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        child_json = Path(tmp) / "result.json"
        cmd = [
            sys.executable, "-m", "src.cli", "benchmark", *argv,
            "--profiles", profile, "--child-json", str(child_json),
        ]  # fmt: skip
        if not verify:
            cmd.append("--no-functional")
        err(f"benchmark: starting {profile} in a fresh process")
        child = subprocess.run(cmd, capture_output=args.quiet, text=True)
        if child_json.is_file():
            return json.loads(child_json.read_text())
        detail = (child.stderr or "").strip().splitlines()[-1:] if args.quiet else []
        error = f"process exited with code {child.returncode}"
        return {"profile": profile, "error": ": ".join([error, *detail])}


def _strip_profile_args(argv: Sequence[str]) -> list[str]:
    """argv without --profiles/--verify-profile/--report/--in-process (re-added per child)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ("--profiles", "--verify-profile", "--report"):
            skip = True
            continue
        if arg.split("=")[0] in ("--profiles", "--verify-profile", "--report"):
            continue
        if arg in ("--in-process", "benchmark"):
            continue
        out.append(arg)
    return out


def cmd_benchmark(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    from src import benchmark

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in OPTIMIZATION_PROFILES]
    if unknown or not profiles:
        raise UsageError(f"unknown profile(s): {', '.join(unknown) or '(none)'}")
    if not args.mock:
        check_model_path(args.model)
    verify_profile = args.verify_profile or profiles[0]
    run_functional = not args.no_functional

    if args.child_json:  # internal: one profile, results as JSON for the parent
        if args.quiet:
            err = lambda message: None  # noqa: E731
        result = _run_profile(args, profiles[0], run_functional, err)
        args.child_json.write_text(json.dumps(result))
        return EXIT_OK

    if args.quiet:
        err = lambda message: None  # noqa: E731 — progress and report echo suppressed
    results = []
    child_argv = _strip_profile_args(args.raw_argv)
    for profile in profiles:
        verify = run_functional and profile == verify_profile
        if args.in_process or len(profiles) == 1:
            results.append(_run_profile(args, profile, verify, err))
        else:
            results.append(_run_profile_subprocess(args, child_argv, profile, verify, err))

    perf = {}
    for r in results:
        if "perf" in r:
            perf[r["profile"]] = r["perf"]
        elif not args.no_perf:
            perf[r["profile"]] = {"error": r.get("error", "no result")}
    checks = None
    for r in results:
        if "checks" in r:
            checks = [
                benchmark.Check(c["name"], c["ok"], c["detail"], [Path(p) for p in c["images"]])
                for c in r["checks"]
            ]
        elif r["profile"] == verify_profile and run_functional and "error" in r:
            checks = [benchmark.Check("functional checks", False, r["error"])]
    sheet = None
    if checks:
        sheet = benchmark.contact_sheet(checks, args.output_dir / "functional_contact_sheet.png")
    environment = next((r["environment"] for r in results if r.get("environment")), [])
    name = args.model.name if args.model else "mock.safetensors"
    report = benchmark.render_report(
        family=args.model_family,
        model_name=name,
        environment=[tuple(row) for row in environment],
        profiles=perf,
        checks=checks,
        sheet=sheet,
    )
    report_path = args.report or Path("compat") / benchmark.default_report_name(args.model_family)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    summary = {
        "report": str(report_path.resolve()),
        "contact_sheet": str(sheet.resolve()) if sheet else None,
        "recommended_profile": benchmark.recommend(perf)[0] if perf else None,
        "checks_passed": None if checks is None else sum(c.ok for c in checks),
        "checks_total": None if checks is None else len(checks),
        "results": results,
    }
    report_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    out(str(report_path))
    err(report)
    failed_checks = checks is not None and not all(c.ok for c in checks)
    no_perf = not args.no_perf and not any("error" not in r for r in perf.values())
    return EXIT_FAILURE if failed_checks or no_perf else EXIT_OK


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    value: str
    status: str = "info"  # info | ok | warn | fail
    hint: str = ""


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _try(fn: Callable[[], Any]) -> tuple[bool, str]:
    try:
        fn()
        return True, ""
    except Exception as exc:  # noqa: BLE001 — diagnostics report any failure
        return False, f"{type(exc).__name__}: {exc}".splitlines()[0][:160]


def _gpu_checks(compile_check: bool, real: bool) -> list[Check]:
    checks: list[Check] = []
    need = "fail" if real else "warn"
    if not _module_available("torch"):
        checks.append(
            Check(
                "PyTorch",
                "not installed",
                need,
                "Colab provides torch; locally use --mock or `uv sync --group inference-cpu`",
            )
        )
        return checks

    configure_cuda_allocator()
    torch = _import_torch()
    checks.append(Check("PyTorch", torch.__version__, "ok"))
    cuda = torch.cuda.is_available()
    checks.append(
        Check(
            "CUDA available",
            "yes" if cuda else "no",
            "ok" if cuda else need,
            "" if cuda else "select a GPU runtime (Runtime → Change runtime type → L4)",
        )
    )
    checks.append(Check("Torch CUDA version", str(torch.version.cuda or "none")))
    device = "cuda" if cuda else "cpu"
    if cuda:
        props = torch.cuda.get_device_properties(0)
        is_l4 = "L4" in props.name
        checks.append(
            Check("GPU", props.name, "ok" if is_l4 else "warn", "" if is_l4 else "tuned for L4")
        )
        checks.append(Check("GPU memory", f"{props.total_memory / 2**30:.1f} GiB"))
        ok, why = _try(lambda: torch.zeros(8, dtype=torch.float16, device="cuda") + 1)
        checks.append(Check("FP16 CUDA tensor", "ok" if ok else why, "ok" if ok else "fail"))

    def sdpa() -> None:
        q = torch.zeros(1, 1, 4, 8, device=device)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)

    ok, why = _try(sdpa)
    checks.append(
        Check("SDPA", "available" if ok else f"unavailable ({why})", "ok" if ok else "warn")
    )
    ok, why = _try(
        lambda: torch.zeros(1, 4, 8, 8, device=device).to(memory_format=torch.channels_last)
    )
    checks.append(
        Check(
            "channels_last", "available" if ok else f"unavailable ({why})", "ok" if ok else "warn"
        )
    )
    has_compile = hasattr(torch, "compile")
    checks.append(
        Check(
            "torch.compile",
            "available" if has_compile else "unavailable",
            "ok" if has_compile else "warn",
            "" if has_compile else "optional; baseline will be used",
        )
    )
    if compile_check and has_compile:

        def smoke() -> None:
            fn = torch.compile(lambda x: torch.sin(x) * 2 + 1)
            fn(torch.ones(16, device=device))

        ok, why = _try(smoke)
        checks.append(
            Check(
                "compile smoke check",
                "pass" if ok else f"fail ({why})",
                "ok" if ok else "warn",
                "" if ok else "optional; baseline will be used",
            )
        )
    else:
        checks.append(Check("compile smoke check", "not-run"))
    return checks


def collect_diagnostics(
    *,
    mock: bool,
    model: Path | None = None,
    family: str | None = None,
    compile_check: bool = False,
) -> list[Check]:
    real = not mock
    checks = [Check("Python", platform.python_version())]

    for dist in ("flask", "pillow"):
        version = _version(dist)
        checks.append(
            Check(dist.capitalize(), version or "not installed", "ok" if version else "fail")
        )

    if mock:
        checks.append(Check("Mode", "mock (GPU stack not checked)"))
    else:
        checks.extend(_gpu_checks(compile_check, real))
        for dist in ("diffusers", "transformers", "accelerate", "safetensors"):
            version = _version(dist)
            required = dist in ("diffusers", "transformers")
            status = "ok" if version else ("fail" if required else "warn")
            hint = "" if version else "pip install -r requirements-inference.txt"
            checks.append(Check(dist.capitalize(), version or "not installed", status, hint))
        checks.append(Check("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "(unset)")))

    if family is not None:
        checks.append(Check("model family", family, "ok"))
    if model is not None:
        exists = model.is_file()
        size = f", {model.stat().st_size / 2**30:.2f} GiB" if exists else ""
        good_ext = model.suffix.lower() == ".safetensors"
        checks.append(
            Check(
                "checkpoint exists",
                f"{'yes' if exists else 'no'} ({model}{size})",
                "ok" if exists and good_ext else "fail",
                "" if good_ext else "V1 accepts only .safetensors",
            )
        )
    return checks


MARKS = {"ok": "", "info": "", "warn": "  [warn]", "fail": "  [FAIL]"}


def format_checks(checks: Sequence[Check]) -> list[str]:
    width = max(len(c.name) for c in checks) + 2
    lines = []
    for c in checks:
        line = f"{c.name + ':':<{width}}{c.value}{MARKS[c.status]}"
        if c.hint and c.status in ("warn", "fail"):
            line += f" — {c.hint}"
        lines.append(line)
    return lines


def cmd_doctor(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    checks = collect_diagnostics(
        mock=args.mock,
        model=args.model,
        family=args.model_family,
        compile_check=args.compile_check,
    )
    for line in format_checks(checks):
        out(line)
    failed = [c.name for c in checks if c.status == "fail"]
    if failed:
        err(f"doctor: required capability missing: {', '.join(failed)}")
        return EXIT_FAILURE
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "doctor": cmd_doctor,
    "generate": cmd_generate,
    "benchmark": cmd_benchmark,
    "serve": cmd_serve,
}


def main(
    argv: Sequence[str] | None = None, out: Printer = print, err: Printer | None = None
) -> int:
    err = err or (lambda message: print(message, file=sys.stderr))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.raw_argv = argv
    try:
        return COMMANDS[args.command](args, out, err)
    except (UsageError, ValidationError) as exc:
        err(f"Error: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
