import pytest
from PIL import Image

import src.cli as cli
from src import benchmark
from src.backend import MockBackend

SMALL = dict(steps=2, size=(256, 256), alt_size=(320, 256), log=lambda m: None)


def loaded_mock(**kwargs):
    backend = MockBackend("sd15", **kwargs)
    backend.load()
    return backend


def test_benchmark_profile_measures_everything(tmp_path):
    result = benchmark.benchmark_profile(MockBackend("sd15"), "sd15", tmp_path, **SMALL)
    expected = (
        "load warmup warm_single_1 warm_single_2 warm_single_3 batch_5 batch_10 "
        "new_size_first new_size_second img2img_first img2img_second"
    )
    assert set(result["timings_s"]) == set(expected.split())
    assert result["profile_active"] == "baseline"
    assert result["peak_vram_batches"] is None  # no CUDA on the mock
    assert result["settings"] == "2 steps, 256x256 (new size 320x256)"


def test_benchmark_records_oom_limits(tmp_path):
    result = benchmark.benchmark_profile(
        MockBackend("sd15", max_batch=4), "sd15", tmp_path, **SMALL
    )
    assert result["batch_limits"] == {"txt2img 256x256": 2}


def test_functional_checks_all_pass_on_mock(tmp_path):
    checks = benchmark.functional_checks(loaded_mock(), "sd15", tmp_path, **SMALL)
    assert [c.name for c in checks if not c.ok] == []
    assert len(checks) == 18
    assert sum(c.name.startswith("sampler ") for c in checks) == 6
    with Image.open(benchmark.contact_sheet(checks, tmp_path / "sheet.png")) as img:
        assert img.width > 1000


@pytest.mark.parametrize(
    ("backend", "check", "fragment"),
    [
        ("black", "txt2img x1", "black image"),
        ("failing", "txt2img x10", "Simulated"),
        ("reloading", "repeated singles, no reload", "latencies"),
    ],
)
def test_failures_are_recorded_not_raised(tmp_path, backend, check, fragment):
    class Black(MockBackend):
        def _draw(self, spec, req, source):
            return Image.new("RGB", (req.width, req.height), "black")

    class Reloading(MockBackend):
        def generate(self, *args, **kwargs):
            self.load_count += 1
            return super().generate(*args, **kwargs)

    make = {
        "black": lambda: Black("sd15"),
        "failing": lambda: MockBackend("sd15", fail_after_images=5),
        "reloading": lambda: Reloading("sd15"),
    }[backend]
    instance = make()
    instance.load()
    checks = {c.name: c for c in benchmark.functional_checks(instance, "sd15", tmp_path, **SMALL)}
    assert not checks[check].ok and fragment in checks[check].detail


def perf(median, startup=10.0, active=None, name="x"):
    return {"warm_single_median_s": median, "startup_s": startup, "profile_active": active or name}


BASE = perf(2.0, name="baseline")


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"baseline": BASE}, "baseline"),
        ({"baseline": BASE, "compile": perf(1.5, name="compile")}, "compile"),
        ({"baseline": BASE, "compile": perf(1.95, name="compile")}, "baseline"),
        ({"baseline": BASE, "compile": {"error": "boom"}}, "baseline"),
        ({"baseline": BASE, "compile": perf(1.0, active="baseline")}, "baseline"),
        ({"compile": {"error": "x"}}, None),
    ],
)  # fmt: skip
def test_recommend(results, expected):
    assert benchmark.recommend(results)[0] == expected


def run_cli(*extra):
    out, err = [], []
    return cli.main(["benchmark", "--mock", "--model-family", "sd15", *extra],
                    out=out.append, err=err.append), out, err  # fmt: skip


def test_cli_benchmark_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run_cli("--profiles", "baseline,compile")
    assert code == 0
    report = (tmp_path / out[0]).read_text()
    assert out[0].startswith("compat/benchmark-sd15-")
    assert "| | baseline | compile |" in report and "18/18 passed." in report
    assert "**Recommended default profile:**" in report
    assert (tmp_path / "outputs/benchmark/functional_contact_sheet.png").is_file()


def test_cli_benchmark_without_functional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run_cli("--profiles", "baseline", "--no-functional")
    assert code == 0 and "Functional checks" not in (tmp_path / out[0]).read_text()


def test_cli_benchmark_load_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_backend", lambda args, log: MockBackend("sd15", fail_load=True))
    code, out, _ = run_cli("--profiles", "baseline")
    assert code == 1
    assert (
        "**baseline failed:** RuntimeError: Mock checkpoint failed to load"
        in (tmp_path / out[0]).read_text()
    )


@pytest.mark.parametrize("profiles", ["turbo", ","])
def test_cli_benchmark_rejects_unknown_profiles(profiles):
    code, _, err = run_cli("--profiles", profiles)
    assert code == 2 and "unknown profile" in err[-1]


def test_functional_crash_keeps_profile_timings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def crash(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(benchmark, "functional_checks", crash)
    code, out, _ = run_cli("--profiles", "baseline")
    report = (tmp_path / out[0]).read_text()
    assert code == 1
    assert "| cold load (s) | error |" not in report  # timings survived
    assert "| functional checks | **FAIL** | OSError: disk full |" in report
