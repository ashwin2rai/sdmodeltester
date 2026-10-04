"""Checks for notebooks/colab.ipynb: structure, safety, and an end-to-end run of its cells."""

import ast
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_colab_utils import CKPT, free_port

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "colab.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
         if c["cell_type"] == "code"]  # fmt: skip


def test_notebook_is_minimal_plain_python():
    assert len(CELLS) == 4
    for src in CELLS:
        ast.parse(src)
        assert not re.search(r"^\s*[!%]", src, re.MULTILINE)  # no IPython magics
    text = NOTEBOOK.read_text()
    assert '"outputs": []' in text and '"execution_count": null' in text


def test_no_secrets_and_spec_settings():
    text = NOTEBOOK.read_text()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    settings = CELLS[0]
    for line in ('MODEL_FAMILY = "sdxl"', 'MODEL_URL = ""', 'HF_TOKEN = ""', 'CIVITAI_TOKEN = ""',
                 "PORT = 8000"):  # fmt: skip
        assert line in settings
    for forbidden in ("ngrok", "cloudflared", "--upgrade torch", "xformers", "--mock"):
        assert forbidden not in text
    assert 'cu.get_secret("HF_TOKEN", HF_TOKEN)' in CELLS[2]


def test_notebook_only_uses_existing_helpers():
    from notebooks import colab_utils

    used = set(re.findall(r"\bcu\.([a-z_]+)\(", "\n".join(CELLS)))
    assert used and all(hasattr(colab_utils, name) for name in used)


def fake_remote(tmp_path) -> str:
    """A git repo with the current working tree, standing in for GitHub."""
    remote = tmp_path / "remote"
    ignore = shutil.ignore_patterns("__pycache__")
    for name in ("src", "notebooks"):
        shutil.copytree(REPO_ROOT / name, remote / name, ignore=ignore)
    for name in ("requirements.txt", "requirements-inference.txt"):
        shutil.copy2(REPO_ROOT / name, remote / name)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "snapshot"]):
        subprocess.run([*git, *args], cwd=remote, check=True)
    return f"file://{remote}"


def test_cells_end_to_end(tmp_path, capsys):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "model.safetensors").write_bytes(CKPT)
    files = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=tmp_path / "files")
    )
    threading.Thread(target=files.serve_forever, daemon=True).start()
    repo_dir = tmp_path / "content" / "sdmodeltester"
    port = free_port()

    cells = [src.replace("/content/sdmodeltester", str(repo_dir)) for src in CELLS]
    cells[1] = (
        cells[1]
        .replace("https://github.com/ashwin2rai/sdmodeltester", fake_remote(tmp_path))
        .replace("cu.install_requirements(REPO_DIR)", "pass  # (test: no pip)")
        .replace('"--model-family", MODEL_FAMILY]', '"--model-family", MODEL_FAMILY, "--mock"]')
    )
    ns: dict = {}
    outputs = []

    def run(i):
        exec(compile(cells[i], f"<cell {i + 1}>", "exec"), ns)
        outputs.append(capsys.readouterr().out)

    saved_path = list(sys.path)
    try:
        run(0)
        ns.update(MODEL_URL=f"http://127.0.0.1:{files.server_address[1]}/model.safetensors",
                  PORT=port)  # fmt: skip
        run(1)
        ns["cu"].MIN_CHECKPOINT_BYTES = 100  # the fake checkpoint is tiny
        run(2)
        assert ns["MODEL_PATH"] == repo_dir / "models" / "model.safetensors"
        run(3)  # real `serve`: UI comes up even though this machine can't load the model
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert response.status == 200
        run(3)  # re-running the cell restarts the server
        assert ns["cu"].port_open(port)
    finally:
        if "cu" in ns:
            ns["cu"].stop_server(repo_dir / "server.pid")
        sys.path[:] = saved_path
        sys.modules.pop("colab_utils", None)
        files.shutdown()

    # The notebook is quiet: one ✓ line per setup cell, nothing from the server.
    # (Download progress is a single line overwritten with "\r", as Colab displays it.)
    assert [o.split("\r")[-1].strip() for o in outputs] == [
        "",
        "✓ Installed",
        "✓ model.safetensors",
        f"Open http://127.0.0.1:{port}/",
        f"Open http://127.0.0.1:{port}/",
    ]


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_settings_cell_runs(family):
    ns: dict = {}
    exec(CELLS[0].replace('MODEL_FAMILY = "sdxl"', f'MODEL_FAMILY = "{family}"'), ns)
    assert ns["MODEL_FAMILY"] == family
