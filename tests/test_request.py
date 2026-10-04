from datetime import datetime
from pathlib import Path

import pytest

from src.backend import (
    FAMILY_DEFAULTS,
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
)
from tests.test_seeds import make_request


def test_valid_request_passes():
    validate_request(make_request())
    validate_request(make_request(input_image=Path("x.png"), strength=0.6))


def test_family_defaults():
    assert FAMILY_DEFAULTS["sd15"].width == 512 and FAMILY_DEFAULTS["sd15"].guidance_scale == 7.5
    assert FAMILY_DEFAULTS["sdxl"].width == 1024 and FAMILY_DEFAULTS["sdxl"].guidance_scale == 5.0
    for defaults in FAMILY_DEFAULTS.values():
        assert defaults.sampler in SAMPLER_IDS
        validate_request(
            make_request(
                width=defaults.width,
                height=defaults.height,
                steps=defaults.steps,
                guidance_scale=defaults.guidance_scale,
                sampler=defaults.sampler,
                seed=defaults.seed,
                num_images=defaults.num_images,
            )
        )


def test_check_family():
    assert check_family("sd15") == "sd15"
    assert check_family("sdxl") == "sdxl"
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
        resolve_job(make_request(**overrides))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(512, 512), (513, 512), (515, 512), (516, 520), (1001, 1000), (255, 256), (2051, 2048)],
)
def test_round_dimension(value, expected):
    assert round_dimension(value) == expected


def test_dimensions_are_rounded_with_note():
    job = resolve_job(make_request(width=516, height=1001))
    assert (job.request.width, job.request.height) == (520, 1000)
    assert len(job.corrections) == 2
    assert "516" in job.corrections[0] and "520" in job.corrections[0]


def test_valid_request_has_no_corrections():
    req = make_request()
    assert normalize_request(req) == (req, ())


@pytest.mark.parametrize(
    ("steps", "strength", "expected_steps"),
    [(5, 0.1, 10), (1, 0.5, 2), (2, 0.3, 4), (1, 0.7, 2), (3, 0.01, 100)],
)
def test_img2img_steps_raised_to_one_denoising_step(steps, strength, expected_steps):
    job = resolve_job(make_request(input_image=Path("x.png"), strength=strength, steps=steps))
    assert job.request.steps == expected_steps
    assert int(job.request.steps * job.request.strength) == 1
    assert job.corrections and "steps raised" in job.corrections[0]


def test_img2img_steps_untouched_when_sufficient():
    job = resolve_job(make_request(input_image=Path("x.png"), strength=0.6, steps=25))
    assert job.request.steps == 25
    assert job.corrections == ()


def test_txt2img_low_steps_untouched():
    assert resolve_job(make_request(steps=1)).request.steps == 1


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
        if sampler != "euler_a":  # EulerAncestralDiscreteScheduler has no karras option
            assert overrides["use_karras_sigmas"] == sampler.endswith("_karras"), sampler
