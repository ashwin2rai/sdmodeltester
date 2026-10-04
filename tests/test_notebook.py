"""Checks for notebooks/colab.ipynb: structure, safety, and an end-to-end demo-mode run."""

import ast
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "colab.ipynb"


def load():
    return json.loads(NOTEBOOK.read_text())


def code_cells():
    return ["".join(c["source"]) for c in load()["cells"] if c["cell_type"] == "code"]


def title(src):
    return src.splitlines()[0]


def test_notebook_structure():
    nb = load()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    titles = [title(src) for src in code_cells()]
    assert [t.split("·")[0].replace("# @title", "").strip() for t in titles] == [
        str(i) for i in range(1, 9)
    ]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_code_cells_compile_without_magics():
    for src in code_cells():
        ast.parse(src)  # no IPython !/% magics: everything is plain Python
        assert not re.search(r"^\s*[!%]", src, re.MULTILINE)


def test_no_secrets_and_safe_defaults():
    text = NOTEBOOK.read_text()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert not re.search(r"\b[0-9a-f]{32}\b", text)
    settings = code_cells()[0]
    assert 'HF_TOKEN = ""' in settings and 'CIVITAI_TOKEN = ""' in settings
    assert 'MODEL_FAMILY = "sdxl"' in settings and 'MODEL_SOURCE = ""' in settings
    assert "DEMO_MODE = False" in settings
    assert 'REPO_URL = "https://github.com/ashwin2rai/sdmodeltester"' in settings
    for forbidden in ("ngrok", "cloudflared", "pip install --upgrade torch", "xformers"):
        assert forbidden not in text


def test_notebook_only_uses_existing_helpers():
    from notebooks import colab_utils

    used = set(re.findall(r"\bcu\.([a-z_]+)\(", "\n".join(code_cells())))
    assert used, "notebook should call colab_utils helpers"
    missing = sorted(name for name in used if not hasattr(colab_utils, name))
    assert missing == []


def test_tokens_come_from_secrets_first():
    download = next(src for src in code_cells() if "4 · Download" in title(src))
    assert 'cu.get_secret("HF_TOKEN", HF_TOKEN)' in download
    assert 'cu.get_secret("CIVITAI_TOKEN", CIVITAI_TOKEN)' in download


# ---------------------------------------------------------------------------
# End-to-end in demo mode (no GPU, no model, no network)
# ---------------------------------------------------------------------------


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_fake_remote(tmp_path) -> str:
    """A git repo containing the current working tree, standing in for GitHub."""
    remote = tmp_path / "remote"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("src", "notebooks"):
        shutil.copytree(REPO_ROOT / name, remote / name, ignore=ignore)
    for name in ("requirements.txt", "requirements-inference.txt", "pyproject.toml"):
        shutil.copy2(REPO_ROOT / name, remote / name)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=remote,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "snapshot")
    return f"file://{remote}"


def http_json(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_demo_mode_end_to_end(tmp_path, capsys):
    cells = {title(src).split("·")[0].replace("# @title", "").strip(): src for src in code_cells()}
    ns: dict = {"__name__": "__notebook__"}
    port = free_port()
    outputs = []

    def run(number):
        exec(compile(cells[number], f"<cell {number}>", "exec"), ns)
        outputs.append(capsys.readouterr().out)
        return outputs[-1]

    def wait_ready():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if http_json(f"http://127.0.0.1:{port}/api/status")["backend_state"] == "ready":
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise AssertionError("server never became ready")

    saved_path = list(sys.path)
    try:
        run("1")
        ns.update(
            DEMO_MODE=True,
            PORT=port,
            WORK_DIR=str(tmp_path / "content"),
            REPO_URL=make_fake_remote(tmp_path),
            REPO_REF="main",
        )
        out = run("2")  # clone + preflight (CPU is fine in demo mode)
        assert out.startswith("✓ Code ") and "Python " in out
        repo = Path(ns["REPO_DIR"])
        assert (repo / "src" / "cli.py").is_file()
        ns["cu"].install_requirements = lambda *a, **k: "test"
        assert run("3").strip() == "✓ Installed and checked (torch test untouched)"
        assert run("4").strip() == "✓ Demo mode: no model needed"
        assert ns["MODEL_PATH"] is None
        out = run("5")
        assert out.strip().endswith("✓ UI running")
        wait_ready()

        # use the running server like the UI would
        prompt = "a {crimson | azure} fox wearing goggles"
        job = http_json(f"http://127.0.0.1:{port}/api/queue", {"prompt": prompt, "seed": 987654})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            latest = http_json(f"http://127.0.0.1:{port}/api/status")["latest_completed_job"]
            if latest and latest["id"] == job["job_id"]:
                break
            time.sleep(0.2)
        assert latest and len(latest["outputs"]) == 1

        out = run("6")
        assert (tmp_path / "content" / "sdmodeltester-outputs.zip").is_file()

        run("2")  # re-running the setup cell updates in place
        out = run("5")  # ...and restarting the server replaces the old one
        assert out.strip().endswith("✓ UI running")
        wait_ready()

        ns["PROFILES"] = "baseline"  # keep the demo benchmark short
        cells["7"] = cells["7"].replace('PROFILES = "baseline,compile"', 'PROFILES = "baseline"')
        out = run("7")
        assert "Report:" in out and "18/18 passed." in out
        assert Path(ns["BENCHMARK"]["contact_sheet"]).is_file()
    finally:
        if "cu" in ns and "PID_FILE" in ns:
            outputs.append("")
            exec(compile(cells["8"], "<cell 8>", "exec"), ns)
            outputs.append(capsys.readouterr().out)
        sys.path[:] = saved_path
        sys.modules.pop("colab_utils", None)
    assert outputs[-1].strip() in ("✓ UI stopped", "The UI was not running.")

    # The notebook is "deaf" to the UI: no prompts, server log lines or job details.
    everything = "\n".join(outputs)
    for leaked in ("crimson", "azure", "goggles", "Queued job", "Backend:", "Job 1", "987654"):
        assert leaked not in everything, leaked


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_settings_cell_validates_family(family):
    ns: dict = {}
    src = code_cells()[0].replace('MODEL_FAMILY = "sdxl"', f'MODEL_FAMILY = "{family}"')
    exec(compile(src, "<settings>", "exec"), ns)
    bad = code_cells()[0].replace('MODEL_FAMILY = "sdxl"', 'MODEL_FAMILY = "auto"')
    with pytest.raises(ValueError):
        exec(compile(bad, "<settings>", "exec"), {})
