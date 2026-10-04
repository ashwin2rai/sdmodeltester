"""Real-backend checks that need neither torch nor diffusers."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.backend import (
    OPTIMIZATION_PROFILES,
    SAMPLER_IDS,
    SAMPLER_SCHEDULERS,
    DiffusersBackend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_sampler_has_a_scheduler_mapping():
    assert set(SAMPLER_SCHEDULERS) == set(SAMPLER_IDS)


def test_karras_flags_are_explicit():
    for sampler, (_, overrides) in SAMPLER_SCHEDULERS.items():
        if sampler == "euler_a":
            continue  # EulerAncestralDiscreteScheduler has no karras option
        assert overrides["use_karras_sigmas"] == sampler.endswith("_karras"), sampler


def test_construction_is_lazy_and_validates():
    code = (
        "import sys; from src.backend import DiffusersBackend; "
        "b = DiffusersBackend('sdxl', 'models/x.safetensors'); "
        "assert b.model_name == 'x.safetensors' and b.warmup_size == (1024, 1024); "
        "assert 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
    with pytest.raises(ValueError):
        DiffusersBackend("sd15", "x.safetensors", optimization="turbo")
    with pytest.raises(ValueError):
        DiffusersBackend("sd21", "x.safetensors")
    assert "baseline" in OPTIMIZATION_PROFILES
