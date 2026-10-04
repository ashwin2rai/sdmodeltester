import json
from pathlib import Path

import pytest
from PIL import Image

import src.cli as cli
from src import benchmark
from src.backend import MockBackend


def loaded_mock(**kwargs):
    backend = MockBackend("sd15", **kwargs)
    backend.load()
    backend.warmup()
    return backend


SMALL = dict(steps=2, size=(256, 256), alt_size=(320, 256), log=lambda m: None)


def test_benchmark_profile_measures_everything(tmp_path):
    result = benchmark.benchmark_profile(MockBackend("sd15"), "sd15", tmp_path, **SMALL)
    timings = result["timings_s"]
    for key in (
        "load",
        "warmup",
        "warm_single_1",
        "warm_single_2",
        "warm_single_3",
        "batch_5",
        "batch_10",
        "batch_10_images_per_s",
        "new_size_first",
        "new_size_second",
        "img2img_first",
        "img2img_second",
    ):
        assert key in timings and timings[key] >= 0, key
    assert result["profile_active"] == "baseline"
    assert result["memory"]["batch_10"] is None  # no CUDA on the mock
    assert result["size"] == [256, 256] and result["alt_size"] == [320, 256]
    assert json.dumps(result)  # serialisable for the parent process


def test_benchmark_records_oom_limits(tmp_path):
    backend = MockBackend("sd15", max_batch=4)
    result = benchmark.benchmark_profile(backend, "sd15", tmp_path, **SMALL)
    assert result["batch_limits"] == {"txt2img 256x256": 2}


def test_functional_checks_all_pass_on_mock(tmp_path):
    checks = benchmark.functional_checks(loaded_mock(), "sd15", tmp_path, **SMALL)
    assert [c.name for c in checks if not c.ok] == []
    names = [c.name for c in checks]
    assert len([n for n in names if n.startswith("sampler ")]) == 6
    assert {"txt2img x10", "img2img x10", "fixed seed reproducible", "non-square"} <= set(names)
    sheet = benchmark.contact_sheet(checks, tmp_path / "sheet.png")
    with Image.open(sheet) as img:
        assert img.width > 1000 and img.height > 100


def test_black_images_fail_checks(tmp_path):
    class BlackBackend(MockBackend):
        def _draw(self, spec, req, source):
            return Image.new("RGB", (req.width, req.height), "black")

    backend = BlackBackend("sd15")
    backend.load()
    checks = benchmark.functional_checks(backend, "sd15", tmp_path, **SMALL)
    txt = next(c for c in checks if c.name == "txt2img x1")
    assert not txt.ok and "black image" in txt.detail


def test_failing_check_is_recorded_not_raised(tmp_path):
    backend = loaded_mock(fail_after_images=5)
    checks = benchmark.functional_checks(backend, "sd15", tmp_path, **SMALL)
    ten = next(c for c in checks if c.name == "txt2img x10")
    assert not ten.ok and "Simulated" in ten.detail


def test_reload_detected(tmp_path):
    class Reloading(MockBackend):
        def generate(self, job, output_dir, progress_callback=None):
            self.load_count += 1
            return super().generate(job, output_dir, progress_callback)

    backend = Reloading("sd15")
    backend.load()
    checks = benchmark.functional_checks(backend, "sd15", tmp_path, **SMALL)
    assert not next(c for c in checks if c.name == "repeated singles, no reload").ok


def perf(median, startup=10.0, active=None, name="x"):
    return {"warm_single_median_s": median, "startup_s": startup, "profile_active": active or name}


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"baseline": perf(2.0, name="baseline")}, "baseline"),
        (
            {"baseline": perf(2.0, name="baseline"), "compile": perf(1.5, 60, name="compile")},
            "compile",
        ),
        (
            {"baseline": perf(2.0, name="baseline"), "compile": perf(1.95, name="compile")},
            "baseline",
        ),
        ({"baseline": perf(2.0, name="baseline"), "compile": {"error": "boom"}}, "baseline"),
        # compile fell back to baseline at warm-up: not a real compile result
        (
            {"baseline": perf(2.0, name="baseline"), "compile": perf(1.0, active="baseline")},
            "baseline",
        ),
        ({"compile": {"error": "x"}}, None),
    ],
)
def test_recommend(results, expected):
    choice, reason = benchmark.recommend(results)
    assert choice == expected
    assert reason


def test_render_report_sections(tmp_path):
    profiles = {
        "baseline": benchmark.benchmark_profile(MockBackend("sd15"), "sd15", tmp_path, **SMALL),
        "compile": {"error": "RuntimeError: boom"},
    }
    checks = [benchmark.Check("a", True, "fine"), benchmark.Check("b", False, "x | y")]
    text = benchmark.render_report(
        family="sd15",
        model_name="m.safetensors",
        environment=[("Python", "3.13"), ("GPU", "NVIDIA L4")],
        profiles=profiles,
        checks=checks,
        sheet=tmp_path / "sheet.png",
    )
    assert "# Benchmark: sd15 · m.safetensors" in text
    assert "- GPU: NVIDIA L4" in text
    assert "| | baseline | compile |" in text
    assert "| cold load (s) |" in text and "| error |" in text
    assert "**compile failed:** RuntimeError: boom" in text
    assert "**Recommended default profile:** baseline" in text
    assert "1/2 passed." in text and "| b | **FAIL** | x \\| y |" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def bench_argv(tmp_path, *extra):
    return [
        "benchmark",
        "--mock",
        "--model-family",
        "sd15",
        "--steps",
        "2",
        "--width",
        "256",
        "--height",
        "256",
        "--output-dir",
        str(tmp_path / "out"),
        "--report",
        str(tmp_path / "report.md"),
        *extra,
    ]


def run_cli(argv):
    out, err = [], []
    return cli.main(argv, out=out.append, err=err.append), out, err


def test_cli_benchmark_in_process(tmp_path):
    code, out, _ = run_cli(bench_argv(tmp_path, "--profiles", "baseline,compile", "--in-process"))
    assert code == 0
    report = (tmp_path / "report.md").read_text()
    assert out == [str(tmp_path / "report.md")]
    assert "| | baseline | compile |" in report and "18/18 passed." in report
    data = json.loads((tmp_path / "report.json").read_text())
    assert [r["profile"] for r in data["results"]] == ["baseline", "compile"]
    assert "checks" in data["results"][0] and "checks" not in data["results"][1]
    assert data["contact_sheet"] == str(
        (tmp_path / "out" / "functional_contact_sheet.png").resolve()
    )
    assert Path(data["contact_sheet"]).is_file()
    assert data["report"] == str((tmp_path / "report.md").resolve())
    assert (data["checks_passed"], data["checks_total"]) == (18, 18)
    assert data["recommended_profile"] in ("baseline", "compile")


def test_cli_benchmark_quiet_prints_only_report_path(tmp_path, capfd):
    import subprocess
    import sys

    argv = bench_argv(tmp_path, "--profiles", "baseline,compile", "--quiet")
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", *argv],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path / "report.md")
    assert proc.stderr == ""  # children are quiet too


def test_cli_benchmark_subprocess_per_profile(tmp_path):
    code, _, _ = run_cli(bench_argv(tmp_path, "--profiles", "baseline,compile", "--no-functional"))
    assert code == 0
    report = (tmp_path / "report.md").read_text()
    assert "| | baseline | compile |" in report and "Functional checks" not in report
    assert (tmp_path / "out" / "baseline").is_dir() and (tmp_path / "out" / "compile").is_dir()


def test_cli_benchmark_functional_only(tmp_path):
    code, _, _ = run_cli(bench_argv(tmp_path, "--no-perf"))
    assert code == 0
    report = (tmp_path / "report.md").read_text()
    assert "## Performance" not in report and "18/18 passed." in report


def test_cli_benchmark_load_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_backend",
        lambda args, log, mock_delays=(0, 0): MockBackend("sd15", fail_load=True),
    )
    code, _, _ = run_cli(bench_argv(tmp_path, "--in-process"))  # patch can't reach children
    assert code == 1
    report = (tmp_path / "report.md").read_text()
    assert "**baseline failed:** RuntimeError: Mock checkpoint failed to load" in report


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--profiles", "turbo"], "unknown profile"),
        (["--profiles", ","], "unknown profile"),
    ],
)
def test_cli_benchmark_usage_errors(tmp_path, argv, fragment):
    code, _, err = run_cli(bench_argv(tmp_path, *argv))
    assert code == 2 and fragment in err[-1]


def test_cli_benchmark_real_requires_model(tmp_path):
    code, _, err = run_cli(["benchmark", "--model-family", "sdxl"])
    assert code == 2 and "--model is required" in err[-1]


def test_strip_profile_args():
    argv = [
        "benchmark", "--mock", "--profiles", "a,b", "--report", "r.md",
        "--verify-profile=b", "--in-process", "--steps", "2",
    ]  # fmt: skip
    assert cli._strip_profile_args(argv) == ["--mock", "--steps", "2"]


def test_default_report_name():
    assert benchmark.default_report_name("sdxl").startswith("benchmark-sdxl-")
    assert Path(benchmark.default_report_name("sdxl")).suffix == ".md"
