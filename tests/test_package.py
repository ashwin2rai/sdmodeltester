"""Guards the mock-first contract: the package must import without GPU libraries."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_pull_gpu_stack():
    code = (
        "import sys, src; "
        "bad = [m for m in ('torch', 'diffusers', 'transformers') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
