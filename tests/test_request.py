import json
from datetime import datetime
from pathlib import Path

import pytest

from src.backend import (
    FAMILIES,
    MAX_SEED,
    SAMPLER_IDS,
    SAMPLER_SCHEDULERS,
    ValidationError,
    check_family,
    collision_safe_path,
    min_img2img_steps,
    normalize_request,
    output_filename,
    resolve_job,
    round_dimension,
    validate_request,
    zimage_storage_dtype,
)
from tests.test_seeds import make_request


def test_valid_request_passes():
    validate_request(make_request(), "sd15")
    validate_request(make_request(input_image=Path("x.png"), strength=0.6), "sd15")


def test_family_defaults():
    d = {family: spec.defaults for family, spec in FAMILIES.items()}
    assert d["sd15"].width == 512 and d["sd15"].guidance_scale == 7.5
    assert d["sdxl"].width == 1024 and d["sdxl"].guidance_scale == 5.0
    assert d["zimage"].steps == 9 and d["zimage"].guidance_scale == 0
    for family, defaults in d.items():
        validate_request(
            make_request(
                width=defaults.width,
                height=defaults.height,
                steps=defaults.steps,
                guidance_scale=defaults.guidance_scale,
                sampler=defaults.sampler,
                seed=defaults.seed,
                num_images=defaults.num_images,
            ),
            family,
        )


def test_check_family():
    assert check_family("sd15") == "sd15"
    assert check_family("sdxl") == "sdxl"
    assert check_family("zimage") == "zimage"
    for bad in ("auto", "SDXL", "sd21", ""):
        with pytest.raises(ValidationError):
            check_family(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 128},
        {"width": 251},  # rounds to 248, still below the minimum
        {"height": 4096},
        {"height": 2052},  # rounds to 2056, above the maximum
        {"width": 512.0},
        {"steps": 0},
        {"steps": True},
        {"guidance_scale": -1},
        {"guidance_scale": float("nan")},
        {"sampler": "DPMSolverMultistepScheduler"},
        {"seed": -2},
        {"seed": MAX_SEED + 1},
        {"seed": MAX_SEED, "num_images": 2},
        {"num_images": 0},
        {"num_images": 11},
        {"prompt": "a {red | } cat"},
        {"negative_prompt": "{x}"},
        {"prompt": "x" * 5000},
        {"strength": 0.5},  # strength without input image
        {"input_image": Path("x.png")},  # img2img without strength
        {"input_image": Path("x.png"), "strength": 0},
        {"input_image": Path("x.png"), "strength": 1.5},
        {"input_image": Path("x.png"), "strength": 0.001},  # would need > MAX_STEPS
    ],
)
def test_invalid_requests_rejected(overrides):
    with pytest.raises(ValidationError):
        resolve_job(make_request(**overrides), family="sd15")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(512, 512), (513, 512), (515, 512), (516, 520), (1001, 1000), (255, 256), (2051, 2048)],
)
def test_round_dimension(value, expected):
    assert round_dimension(value) == expected


def test_dimensions_are_rounded_with_note():
    job = resolve_job(make_request(width=516, height=1001), family="sd15")
    assert (job.request.width, job.request.height) == (520, 1000)
    assert len(job.corrections) == 2
    assert "516" in job.corrections[0] and "520" in job.corrections[0]


def test_valid_request_has_no_corrections():
    req = make_request()
    assert normalize_request(req, "sd15") == (req, ())


@pytest.mark.parametrize(
    ("steps", "strength", "expected_steps"),
    [(5, 0.1, 10), (1, 0.5, 2), (2, 0.3, 4), (1, 0.7, 2), (3, 0.01, 100)],
)
def test_img2img_steps_raised_to_one_denoising_step(steps, strength, expected_steps):
    job = resolve_job(
        make_request(input_image=Path("x.png"), strength=strength, steps=steps), family="sd15"
    )
    assert job.request.steps == expected_steps
    assert int(job.request.steps * job.request.strength) == 1
    assert job.corrections and "steps raised" in job.corrections[0]


def test_img2img_steps_untouched_when_sufficient():
    job = resolve_job(
        make_request(input_image=Path("x.png"), strength=0.6, steps=25), family="sd15"
    )
    assert job.request.steps == 25
    assert job.corrections == ()


def test_txt2img_low_steps_untouched():
    assert resolve_job(make_request(steps=1), family="sd15").request.steps == 1


@pytest.mark.parametrize("strength", [0.1, 0.3, 0.7, 0.07, 0.013, 1 / 3, 0.29, 0.57])
def test_min_img2img_steps_is_minimal(strength):
    steps = min_img2img_steps(strength)
    assert int(steps * strength) >= 1
    assert steps == 1 or int((steps - 1) * strength) < 1


def test_mode():
    assert make_request().mode == "txt2img"
    assert make_request(input_image=Path("x.png"), strength=0.6).mode == "img2img"


def test_output_filename_format():
    now = datetime(2026, 10, 3, 16, 24, 55)
    assert output_filename(7, 0, 123, now) == "20261003_162455_job0007_img01_seed123.png"
    assert output_filename(7, 9, 132, now) == "20261003_162455_job0007_img10_seed132.png"


def test_collision_safe_path(tmp_path):
    assert collision_safe_path(tmp_path, "a.png") == tmp_path / "a.png"
    (tmp_path / "a.png").touch()
    assert collision_safe_path(tmp_path, "a.png") == tmp_path / "a_1.png"
    (tmp_path / "a_1.png").touch()
    assert collision_safe_path(tmp_path, "a.png") == tmp_path / "a_2.png"


def test_every_sampler_has_an_explicit_scheduler_mapping():
    assert set(SAMPLER_SCHEDULERS) == set(SAMPLER_IDS)
    for sampler, (_, overrides) in SAMPLER_SCHEDULERS.items():
        if sampler not in ("euler_a", "flow_euler"):  # schedulers without a karras option
            assert overrides["use_karras_sigmas"] == sampler.endswith("_karras"), sampler


# ---------------------------------------------------------------------------
# Z-Image family
# ---------------------------------------------------------------------------


def zimage_request(**overrides):
    fields = dict(width=1024, height=1024, steps=9, guidance_scale=0.0, sampler="flow_euler")
    return make_request(**{**fields, **overrides})


def test_samplers_are_checked_per_family():
    resolve_job(zimage_request(), family="zimage")
    with pytest.raises(ValidationError, match="not available for zimage"):
        resolve_job(zimage_request(sampler="euler"), family="zimage")
    with pytest.raises(ValidationError, match="not available for sd15"):
        resolve_job(make_request(sampler="flow_euler"), family="sd15")
    assert "flow_euler" not in FAMILIES["sdxl"].samplers


def test_zimage_sizes_round_to_multiples_of_16():
    job = resolve_job(zimage_request(width=1000, height=1032), family="zimage")
    assert (job.request.width, job.request.height) == (1008, 1040)
    assert "multiple of 16" in job.corrections[0]
    with pytest.raises(ValidationError, match="multiple of 16"):
        validate_request(zimage_request(width=1032), family="zimage")
    assert resolve_job(make_request(width=1032), family="sdxl").request.width == 1032


def test_zimage_prompts_resolve_without_weights():
    req = zimage_request(prompt="a {(red:1.3) | (red:0.8)} cat, (snow:1.2)", negative_prompt="")
    job = resolve_job(req, family="zimage")
    assert job.images[0].prompt == "a red cat, snow"
    assert job.corrections == ("prompt weights like (text:1.2) are ignored for zimage",)
    sd_job = resolve_job(make_request(prompt="a (red:1.3) cat", seed=1), family="sd15")
    assert sd_job.images[0].prompt == "a (red:1.3) cat" and sd_job.corrections == ()


@pytest.mark.parametrize(
    ("negative_prompt", "guidance_scale", "noted"),
    [("blurry", 0.0, True), ("", 0.0, False), ("blurry", 2.0, False)],
)
def test_negative_prompt_note_only_without_cfg(negative_prompt, guidance_scale, noted):
    req = zimage_request(negative_prompt=negative_prompt, guidance_scale=guidance_scale)
    notes = resolve_job(req, family="zimage").corrections
    assert notes == (("negative prompt is ignored at CFG 0",) if noted else ())


# ---------------------------------------------------------------------------
# Z-Image checkpoint header inspection (no torch: header written by hand)
# ---------------------------------------------------------------------------


def write_safetensors(path, tensors):
    """Header-only .safetensors with ``{name: dtype}`` (data offsets are irrelevant here)."""
    header = {
        name: {"dtype": dt, "shape": [1], "data_offsets": [0, 1]} for name, dt in tensors.items()
    }
    data = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    path.write_bytes(len(data).to_bytes(8, "little") + data + b"\0")
    return path


ZKEY = "layers.0.adaLN_modulation.0.weight"


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ({ZKEY: "BF16", "x_embedder.weight": "BF16"}, None),
        ({ZKEY: "F8_E4M3", "x_embedder.weight": "F8_E4M3"}, "float8_e4m3fn"),
        ({ZKEY: "F8_E5M2", "layers.0.attention_norm1.weight": "BF16"}, "float8_e5m2"),
        ({"model.diffusion_model." + ZKEY: "F8_E4M3"}, "float8_e4m3fn"),
    ],
)
def test_zimage_storage_dtype(tmp_path, dtypes, expected):
    assert zimage_storage_dtype(write_safetensors(tmp_path / "z.safetensors", dtypes)) == expected


@pytest.mark.parametrize(
    ("dtypes", "fragment"),
    [
        ({"model.diffusion_model.input_blocks.0.0.weight": "F16"}, "not a Z-Image"),
        ({ZKEY: "F8_E4M3", "layers.0.feed_forward.w1._weight_scale": "F32"}, "scaled/quantized"),
        ({ZKEY: "F8_E4M3", "layers.0.feed_forward.w1.scale_weight": "F32"}, "scaled/quantized"),
        ({ZKEY: "I8"}, "unsupported weight types"),
        ({ZKEY: "F8_E4M3", "x_embedder.weight": "F8_E5M2"}, "unsupported weight types"),
    ],
)
def test_zimage_storage_dtype_rejects(tmp_path, dtypes, fragment):
    with pytest.raises(ValueError, match=fragment):
        zimage_storage_dtype(write_safetensors(tmp_path / "z.safetensors", dtypes))


def test_zimage_storage_dtype_rejects_non_safetensors(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="not a valid"):
        zimage_storage_dtype(bad)
