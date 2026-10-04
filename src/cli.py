"""Command-line interface: ``doctor``, ``generate``, ``serve``, ``benchmark``.

    python -m src.cli doctor [--mock] [--model PATH --model-family sd15|sdxl] [--compile-check]
    python -m src.cli generate --model-family sdxl --model models/x.safetensors --prompt "..."
    python -m src.cli generate --mock --model-family sd15 --prompt "a {white | black} cat"
    python -m src.cli serve --model-family sdxl --model models/x.safetensors --port 8000
    python -m src.cli benchmark --model-family sdxl --model models/x.safetensors

Mock mode never imports torch. Real mode sets the CUDA allocator config before torch is
first imported (SPEC §8).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
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
    Backend,
    DiffusersBackend,
    GenerationError,
    GenerationRequest,
    MockBackend,
    ValidationError,
    configure_cuda_allocator,
    describe_request,
    resolve_job,
)

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2
BENCHMARK_DIR = Path("outputs/benchmark")

Printer = Callable[[str], None]


class UsageError(Exception):
    """Bad combination of arguments (exit code 2)."""


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str, *, family_required: bool = True, backend: bool = True):
        p = sub.add_parser(name, help=help)
        p.add_argument("--model-family", choices=MODEL_FAMILIES, required=family_required)
        p.add_argument("--model", type=Path, help="single-file .safetensors checkpoint")
        p.add_argument("--mock", action="store_true", help="placeholder backend: no GPU/torch")
        if backend:
            p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
            p.add_argument("--dtype", default="float16", help=argparse.SUPPRESS)
            p.add_argument(
                "--optimization", choices=OPTIMIZATION_PROFILES, default=DEFAULT_OPTIMIZATION
            )
        return p

    doctor = command("doctor", "report runtime/GPU capabilities", family_required=False,
                     backend=False)  # fmt: skip
    doctor.add_argument(
        "--compile-check", action="store_true", help="run a torch.compile smoke test"
    )

    gen = command("generate", "generate images once and exit")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--cfg", type=float, help="guidance scale")
    gen.add_argument("--sampler", choices=SAMPLER_IDS)
    gen.add_argument("--seed", type=int, default=-1, help="-1 = random per image")
    gen.add_argument("--images", type=int, default=1, help="1-10")
    gen.add_argument("--image", type=Path, help="input image for image-to-image")
    gen.add_argument("--strength", type=float, help="img2img strength")
    gen.add_argument("--output-dir", type=Path, default=Path("outputs"))

    serve = command("serve", "run the HTTP server and browser UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--inputs-dir", type=Path, default=Path("inputs"))
    serve.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    serve.add_argument("--mock-load-seconds", type=float, default=2.0)
    serve.add_argument("--mock-step-seconds", type=float, default=0.05)

    bench = command("benchmark", "Phase 8: compare optimization profiles + functional checks")
    bench.add_argument("--profiles", default="baseline,compile", help="comma-separated")
    bench.add_argument("--no-functional", action="store_true", help="skip functional checks")
    return parser


def build_backend(args: argparse.Namespace, log: Printer, **mock_options: float) -> Backend:
    if args.mock:
        name = args.model.name if args.model else "mock.safetensors"
        return MockBackend(args.model_family, name, **mock_options)
    if args.model is None:
        raise UsageError("--model is required unless --mock is given")
    if args.model.suffix.lower() != ".safetensors":
        raise UsageError(f"--model must be a .safetensors file (got {args.model.name})")
    if not args.model.is_file():
        raise UsageError(f"checkpoint not found: {args.model}")
    configure_cuda_allocator()  # must precede the first torch import
    return DiffusersBackend(
        args.model_family,
        args.model,
        device=args.device,
        dtype=args.dtype,
        optimization=args.optimization,
        log=log,
    )


def build_request(args: argparse.Namespace) -> tuple[GenerationRequest, list[str]]:
    """Fill unset knobs from family defaults; returns the request and CLI notes."""
    d = FAMILY_DEFAULTS[args.model_family]
    notes, strength = [], None
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
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    request, notes = build_request(args)
    job = resolve_job(request, job_id=1)
    for note in (*notes, *job.corrections):
        err(f"Note: {note}")

    backend = build_backend(args, err)
    try:
        backend.load(lambda state, message: err(f"Backend: {message}"))
        if args.optimization != "baseline":
            backend.warmup(lambda state, message: err(f"Backend: {message}"))
    except Exception as exc:  # noqa: BLE001 — report load failures concisely
        err(f"Error: backend failed to load: {type(exc).__name__}: {exc}")
        return EXIT_FAILURE

    for line in describe_request(job):
        err(f"Job 1: {line}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_decile = [-1]

    def progress(fraction: float, message: str) -> None:
        if int(fraction * 10) != last_decile[0]:  # report roughly every 10%
            last_decile[0] = int(fraction * 10)
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


def cmd_serve(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    from src import server

    backend = build_backend(
        args, err, load_seconds=args.mock_load_seconds, step_seconds=args.mock_step_seconds
    )
    server.serve(
        backend,
        host=args.host,
        port=args.port,
        inputs_dir=args.inputs_dir,
        outputs_dir=args.outputs_dir,
    )
    return EXIT_OK


def cmd_benchmark(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    """Profiles run one after another in this process, each with a fresh backend; the
    first one also runs the functional checks. Report: ``compat/benchmark-<family>-<date>.md``.
    """
    from src import benchmark

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in OPTIMIZATION_PROFILES]
    if unknown or not profiles:
        raise UsageError(f"unknown profile(s): {', '.join(unknown) or '(none)'}")
    environment = [
        (c.name, c.value)
        for c in collect_diagnostics(mock=args.mock, model=args.model, family=args.model_family)
    ]
    perf: dict[str, dict] = {}
    checks = None
    for profile in profiles:
        backend = build_backend(argparse.Namespace(**{**vars(args), "optimization": profile}), err)
        try:
            perf[profile] = benchmark.benchmark_profile(
                backend, args.model_family, BENCHMARK_DIR / profile, profile=profile, log=err
            )
        except Exception as exc:  # noqa: BLE001 — a crashed profile is a result
            perf[profile] = {"error": f"{type(exc).__name__}: {exc}"}
        if checks is None and not args.no_functional and "error" not in perf[profile]:
            try:
                checks = benchmark.functional_checks(
                    backend, args.model_family, BENCHMARK_DIR / "functional", log=err
                )
            except Exception as exc:  # noqa: BLE001 — keep this profile's timings
                checks = [
                    benchmark.Check("functional checks", False, f"{type(exc).__name__}: {exc}")
                ]
        del backend
        gc.collect()  # release the previous pipeline before loading the next one
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()

    sheet = benchmark.contact_sheet(checks, BENCHMARK_DIR / "functional_contact_sheet.png")
    report = benchmark.render_report(
        family=args.model_family,
        model_name=args.model.name if args.model else "mock.safetensors",
        environment=environment,
        profiles=perf,
        checks=checks,
        sheet=sheet,
    )
    path = Path("compat") / benchmark.default_report_name(args.model_family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    out(str(path))
    err(report)
    failed = (checks is not None and not all(c.ok for c in checks)) or all(
        "error" in r for r in perf.values()
    )
    return EXIT_FAILURE if failed else EXIT_OK


# ---------------------------------------------------------------------------
# doctor (SPEC §11)
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


def _package(dist: str, missing: str, hint: str = "") -> Check:
    version = _version(dist)
    return Check(dist.capitalize(), version or "not installed", "ok" if version else missing, hint)


def _probe(name: str, fn: Callable[[], Any], missing: str = "warn", hint: str = "") -> Check:
    """Run a capability check; any exception means the capability is unavailable."""
    try:
        fn()
        return Check(name, "ok", "ok")
    except Exception as exc:  # noqa: BLE001
        why = f"{type(exc).__name__}: {exc}".splitlines()[0][:160]
        return Check(name, f"failed ({why})", missing, hint)


def _gpu_checks(compile_check: bool) -> list[Check]:
    if not _module_available("torch"):
        return [
            Check("PyTorch", "not installed", "fail", "Colab provides torch; locally use --mock")
        ]
    configure_cuda_allocator()
    torch = _import_torch()
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    checks = [
        Check("PyTorch", torch.__version__, "ok"),
        Check("CUDA available", "yes" if cuda else "no", "ok" if cuda else "fail",
              "select a GPU runtime (Runtime → Change runtime type → L4)"),
        Check("Torch CUDA version", str(torch.version.cuda or "none")),
    ]  # fmt: skip
    if cuda:
        props = torch.cuda.get_device_properties(0)
        checks += [
            Check("GPU", props.name, "ok" if "L4" in props.name else "warn", "tuned for L4"),
            Check("GPU memory", f"{props.total_memory / 2**30:.1f} GiB"),
            _probe("FP16 CUDA tensor", lambda: torch.zeros(8, dtype=torch.float16, device="cuda"),
                   "fail"),
        ]  # fmt: skip
    q = torch.zeros(1, 1, 4, 8, device=device)  # also a valid 4-D tensor for channels_last
    optional = "optional; baseline will be used"
    checks += [
        _probe("SDPA", lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q)),
        _probe("channels_last", lambda: q.to(memory_format=torch.channels_last)),
        Check("torch.compile", "available" if hasattr(torch, "compile") else "unavailable",
              "ok" if hasattr(torch, "compile") else "warn", optional),
        _probe("compile smoke check",
               lambda: torch.compile(lambda x: torch.sin(x) + 1)(torch.ones(4, device=device)),
               hint=optional)
        if compile_check else Check("compile smoke check", "not-run"),
    ]  # fmt: skip
    return checks


def collect_diagnostics(
    *, mock: bool, model: Path | None = None, family: str | None = None, compile_check: bool = False
) -> list[Check]:
    checks = [Check("Python", platform.python_version())]
    checks += [_package(dist, "fail") for dist in ("flask", "pillow")]
    if mock:
        checks.append(Check("Mode", "mock (GPU stack not checked)"))
    else:
        checks += _gpu_checks(compile_check)
        hint = "pip install -r requirements-inference.txt"
        checks += [_package(d, "fail", hint) for d in ("diffusers", "transformers")]
        checks += [_package(d, "warn", hint) for d in ("accelerate", "safetensors")]
        checks.append(Check("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "(unset)")))
    if family is not None:
        checks.append(Check("model family", family, "ok"))
    if model is not None:
        ok = model.is_file() and model.suffix.lower() == ".safetensors"
        exists = "yes" if model.is_file() else "no"
        hint = "V1 needs an existing .safetensors file"
        checks.append(
            Check("checkpoint exists", f"{exists} ({model})", "ok" if ok else "fail", hint)
        )
    return checks


def format_checks(checks: Sequence[Check]) -> list[str]:
    width = max(len(c.name) for c in checks) + 2
    marks = {"warn": "  [warn]", "fail": "  [FAIL]"}
    return [
        f"{c.name + ':':<{width}}{c.value}{marks.get(c.status, '')}"
        + (f" — {c.hint}" if c.hint and c.status in marks else "")
        for c in checks
    ]


def cmd_doctor(args: argparse.Namespace, out: Printer, err: Printer) -> int:
    checks = collect_diagnostics(
        mock=args.mock, model=args.model, family=args.model_family, compile_check=args.compile_check
    )
    for line in format_checks(checks):
        out(line)
    failed = [c.name for c in checks if c.status == "fail"]
    if failed:
        err(f"doctor: required capability missing: {', '.join(failed)}")
        return EXIT_FAILURE
    return EXIT_OK


COMMANDS = {
    "doctor": cmd_doctor,
    "generate": cmd_generate,
    "serve": cmd_serve,
    "benchmark": cmd_benchmark,
}


def main(
    argv: Sequence[str] | None = None, out: Printer = print, err: Printer | None = None
) -> int:
    err = err or (lambda message: print(message, file=sys.stderr))
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, out, err)
    except (UsageError, ValidationError) as exc:
        err(f"Error: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
