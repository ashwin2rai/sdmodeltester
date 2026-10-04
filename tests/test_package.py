"""Guards the mock-first contract: mock/CLI/server paths never import the GPU stack."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_paths_never_import_gpu_stack(tmp_path):
    quiet = "out=lambda m: None, err=lambda m: None"
    generate = ["generate", "--mock", "--model-family", "sd15", "--prompt", "x", "--steps", "2",
                "--output-dir", str(tmp_path)]  # fmt: skip
    code = f"""
import sys
import src.benchmark, src.server
from src.backend import DiffusersBackend
from src.cli import main
DiffusersBackend("sdxl", "x.safetensors")  # construction is lazy
assert main(["doctor", "--mock"], {quiet}) == 0
assert main({generate!r}, {quiet}) == 0
loaded = [m for m in ("torch", "diffusers", "transformers") if m in sys.modules]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
