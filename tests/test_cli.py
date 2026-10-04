import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import src.cli as cli
from src.backend import DiffusersBackend, MockBackend

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(argv):
    out, err = [], []
    code = cli.main(argv, out=out.append, err=err.append)
    return code, out, err


def gen_args(tmp_path, *extra):
    return [
        "generate",
        "--mock",
        "--model-family",
        "sd15",
        "--output-dir",
        str(tmp_path),
        *extra,
    ]


# ---------------------------------------------------------------------------
# Parsing / request building
# ---------------------------------------------------------------------------


def test_family_is_required_and_has_no_auto():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--mock", "--prompt", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--mock", "--model-family", "auto", "--prompt", "x"])


@pytest.mark.parametrize(("family", "size", "cfg"), [("sd15", 512, 7.5), ("sdxl", 1024, 5.0)])
def test_request_uses_family_defaults(family, size, cfg):
    args = cli.build_parser().parse_args(
        ["generate", "--mock", "--model-family", family, "--prompt", "x"]
    )
    req, notes = cli.build_request(args)
    assert (req.width, req.height, req.guidance_scale, req.steps) == (size, size, cfg, 25)
    assert req.sampler == "dpmpp_2m_sde_karras" and req.seed == -1 and req.num_images == 1
    assert req.input_image is None and req.strength is None and notes == []


def test_strength_ignored_without_image():
    args = cli.build_parser().parse_args(
        ["generate", "--mock", "--model-family", "sd15", "--prompt", "x", "--strength", "0.3"]
    )
    req, notes = cli.build_request(args)
    assert req.strength is None
    assert "ignored" in notes[0]


def test_image_gets_default_strength(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(src)
    args = cli.build_parser().parse_args(
        ["generate", "--mock", "--model-family", "sdxl", "--prompt", "x", "--image", str(src)]
    )
    req, _ = cli.build_request(args)
    assert req.input_image == src and req.strength == 0.6


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_mock_writes_outputs(tmp_path):
    code, out, err = run_cli(
        gen_args(tmp_path, "--prompt", "a {white | black} cat", "--seed", "123", "--images", "3")
    )
    assert code == 0
    assert len(out) == 3
    assert [Path(p).name.split("_")[-1] for p in out] == [
        "seed123.png",
        "seed124.png",
        "seed125.png",
    ]
    assert all(Path(p).is_file() for p in out)
    summary = next(line for line in err if line.startswith("Job 1:"))
    assert "txt2img" in summary and "seeds 123, 124, 125" in summary
    assert any(line.startswith("Done in") for line in err)


def test_generate_mock_img2img(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (300, 200), "green").save(src)
    out_dir = tmp_path / "out"
    code, out, err = run_cli(
        [
            *gen_args(out_dir, "--prompt", "x"),
            "--image",
            str(src),
            "--strength",
            "0.1",
            "--steps",
            "5",
        ]
    )
    assert code == 0
    assert any("steps raised from 5 to 10" in line for line in err)
    summary = next(line for line in err if line.startswith("Job 1:"))
    assert "img2img" in summary and "strength 0.1" in summary


def test_generate_reports_corrections(tmp_path):
    code, _, err = run_cli(gen_args(tmp_path, "--prompt", "x", "--width", "516"))
    assert code == 0
    assert any("width 516 rounded to 520" in line for line in err)


def test_generate_progress_is_throttled(tmp_path):
    code, _, err = run_cli(gen_args(tmp_path, "--prompt", "x", "--steps", "100"))
    assert code == 0
    progress = [line for line in err if "denoising" in line]
    assert 5 <= len(progress) <= 12
    assert progress[-1].strip() == "denoising 100/100 (100%)"


@pytest.mark.parametrize(
    "extra",
    [
        ["--prompt", "a {red} cat"],
        ["--prompt", "x", "--images", "11"],
        ["--prompt", "x", "--width", "4096"],
        ["--prompt", "x", "--image", "missing.png"],
    ],
)
def test_generate_usage_errors_exit_2(tmp_path, extra):
    code, out, err = run_cli(gen_args(tmp_path, *extra))
    assert code == 2 and out == []
    assert err[-1].startswith("Error:")


def test_real_generate_requires_valid_model(tmp_path):
    base = ["generate", "--model-family", "sd15", "--prompt", "x"]
    code, _, err = run_cli(base)
    assert code == 2 and "--model is required" in err[-1]
    wrong = tmp_path / "model.ckpt"
    wrong.touch()
    code, _, err = run_cli([*base, "--model", str(wrong)])
    assert code == 2 and ".safetensors" in err[-1]
    code, _, err = run_cli([*base, "--model", str(tmp_path / "nope.safetensors")])
    assert code == 2 and "not found" in err[-1]


def test_generate_failure_exits_1_and_lists_completed(tmp_path, monkeypatch):
    def failing_backend(args, log, **kwargs):
        return MockBackend(args.model_family, fail_after_images=2)

    monkeypatch.setattr(cli, "build_backend", failing_backend)
    code, out, err = run_cli(gen_args(tmp_path, "--prompt", "x", "--images", "4"))
    assert code == 1 and out == []
    assert sum("completed before failure" in line for line in err) == 2


def test_generate_load_failure_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_backend",
        lambda args, log, **kwargs: MockBackend("sd15", fail_load=True),
    )
    code, _, err = run_cli(gen_args(tmp_path, "--prompt", "x"))
    assert code == 1 and "failed to load" in err[-1]


def test_build_backend_real_is_configured_lazily(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    model = tmp_path / "m.safetensors"
    model.touch()
    args = cli.build_parser().parse_args(
        ["serve", "--model-family", "sdxl", "--model", str(model), "--optimization", "compile"]
    )
    backend = cli.build_backend(args, print)
    assert isinstance(backend, DiffusersBackend)
    assert backend.optimization == "compile" and backend.device == "cuda"
    assert backend.model_name == "m.safetensors"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_build_backend_mock_delays():
    args = cli.build_parser().parse_args(
        ["serve", "--mock", "--model-family", "sd15", "--mock-step-seconds", "0.2"]
    )
    backend = cli.build_backend(args, print, load_seconds=1.0, step_seconds=0.2)
    assert isinstance(backend, MockBackend)
    assert (backend.load_seconds, backend.step_seconds) == (1.0, 0.2)
    assert backend.model_name == "mock.safetensors"


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def by_name(checks):
    return {c.name: c for c in checks}


def test_doctor_real_without_torch_fails(monkeypatch):
    monkeypatch.setattr(cli, "_module_available", lambda name: False)
    monkeypatch.setattr(cli, "_version", lambda dist: None)
    code, out, err = run_cli(["doctor"])
    assert code == 1
    assert any(line.startswith("PyTorch:") and "[FAIL]" in line for line in out)
    assert "PyTorch" in err[-1] and "Diffusers" in err[-1]


def fake_torch(gpu_name="NVIDIA L4", cuda=True):
    props = SimpleNamespace(name=gpu_name, total_memory=24 * 2**30)
    return SimpleNamespace(
        __version__="9.9.9",
        version=SimpleNamespace(cuda="12.8" if cuda else None),
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_properties=lambda i: props),
        float16="float16",
        channels_last="channels_last",
        zeros=lambda *a, **k: 0,
        sin=lambda x: x,
        ones=lambda *a, **k: 1,
        compile=lambda fn: fn,
        nn=SimpleNamespace(
            functional=SimpleNamespace(scaled_dot_product_attention=lambda q, k, v: q)
        ),
    )


@pytest.fixture
def gpu_env(monkeypatch):
    def install(**kwargs):
        torch = fake_torch(**kwargs)
        monkeypatch.setattr(cli, "_module_available", lambda name: True)
        monkeypatch.setattr(cli, "_version", lambda dist: "1.0")
        monkeypatch.setattr(cli, "_import_torch", lambda: torch)

    return install


def test_doctor_l4_passes(gpu_env):
    gpu_env()
    checks = by_name(cli.collect_diagnostics(mock=False, compile_check=True))
    assert checks["GPU"].value == "NVIDIA L4" and checks["GPU"].status == "ok"
    assert checks["GPU memory"].value == "24.0 GiB"
    assert checks["CUDA available"].status == "ok"
    assert checks["compile smoke check"].value == "ok"
    assert not [c for c in checks.values() if c.status == "fail"]


def test_doctor_other_gpu_only_warns(gpu_env):
    gpu_env(gpu_name="Tesla T4")
    checks = by_name(cli.collect_diagnostics(mock=False))
    assert checks["GPU"].status == "warn"
    assert checks["compile smoke check"].value == "not-run"
    assert not [c for c in checks.values() if c.status == "fail"]


def test_doctor_no_cuda_fails_in_real_mode(gpu_env):
    gpu_env(cuda=False)
    code, out, err = run_cli(["doctor"])
    assert code == 1
    assert "CUDA available" in err[-1]
    assert not any(line.startswith("GPU:") for line in out)


def test_doctor_checkpoint_checks(gpu_env, tmp_path):
    gpu_env()
    good = tmp_path / "m.safetensors"
    good.write_bytes(b"x")
    code, out, _ = run_cli(["doctor", "--model", str(good), "--model-family", "sdxl"])
    assert code == 0
    assert any(line.startswith("checkpoint exists:") and "yes" in line for line in out)
    assert any(line.startswith("model family:") and "sdxl" in line for line in out)

    code, _, err = run_cli(["doctor", "--model", str(tmp_path / "missing.safetensors")])
    assert code == 1 and "checkpoint exists" in err[-1]

    bad = tmp_path / "m.ckpt"
    bad.write_bytes(b"x")
    code, out, _ = run_cli(["doctor", "--model", str(bad)])
    assert code == 1 and any(".safetensors" in line for line in out)
