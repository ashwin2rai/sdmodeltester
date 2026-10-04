"""Helpers for notebooks/colab.ipynb: install, checkpoint download, server launch.

Notebook-only operational code (SPEC §21.1). The app itself is only driven through
``python -m src.cli``; nothing here imports ``src``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

MIN_CHECKPOINT_BYTES = 500 * 2**20  # anything smaller is a LoRA/embedding, not a checkpoint


class DownloadError(RuntimeError):
    pass


def get_secret(name: str, fallback: str = "") -> str:
    """Colab Secrets (🔑 sidebar) first, then the form value, then the environment."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get(name)
    except Exception:  # noqa: BLE001 — not in Colab, no such secret, or access denied
        value = None
    return (value or fallback or os.environ.get(name, "")).strip()


def run(cmd: list[str], what: str, cwd: Path | None = None) -> None:
    """Quiet on success, full output on failure."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{result.stdout[-3000:]}{result.stderr[-3000:]}")


def install_requirements(repo_dir: Path) -> None:
    """Install app-level deps on top of Colab's stack; Colab's torch must stay untouched."""

    def torch_version() -> str | None:
        try:
            return version("torch")
        except PackageNotFoundError:
            return None

    before = torch_version()
    requirements = ["-r", str(repo_dir / "requirements.txt"),
                    "-r", str(repo_dir / "requirements-inference.txt")]  # fmt: skip
    run([sys.executable, "-m", "pip", "install", "-q", *requirements], "pip install")
    if torch_version() != before:
        print(f"⚠️  pip changed torch {before} → {torch_version()}; restart the runtime.")


# ---------------------------------------------------------------------------
# Checkpoint download (SPEC §21.5)
# ---------------------------------------------------------------------------


def fetch_checkpoint(url: str, models_dir: Path, *, hf_token: str = "", civitai_token: str = ""):
    """Hugging Face file link, Civitai version link, or direct URL -> validated checkpoint."""
    url = url.strip()
    host = urlparse(url).hostname or ""
    if host in ("huggingface.co", "hf.co"):
        path = _hf_download(url, models_dir, hf_token)
    elif "civitai" in host:
        path = download(civitai_download_url(url), models_dir, civitai_token)
    elif host:
        path = download(url, models_dir)
    else:
        raise DownloadError("MODEL_URL must be a Hugging Face or Civitai link")
    validate_checkpoint(path)
    return path


def _hf_download(url: str, models_dir: Path, token: str) -> Path:
    """``https://huggingface.co/{org}/{repo}/(blob|resolve)/{revision}/{file}``."""
    parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
    if len(parts) < 5 or parts[2] not in ("blob", "resolve"):
        raise DownloadError("Hugging Face link must point at a .safetensors file")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=f"{parts[0]}/{parts[1]}", revision=parts[3], filename="/".join(parts[4:]),
        token=token or None, local_dir=models_dir,
    ))  # fmt: skip


def civitai_download_url(url: str) -> str:
    """Model page ``…/models/1?modelVersionId=2`` or ``…/api/download/models/2`` -> download URL."""
    parsed = urlparse(url)
    if "/api/download/models/" in parsed.path:
        return url
    if version_id := parse_qs(parsed.query).get("modelVersionId"):
        return (
            f"https://civitai.com/api/download/models/{version_id[0]}?type=Model&format=SafeTensor"
        )
    raise DownloadError(
        "Civitai link must include the version: open the model page, pick the version, and "
        "copy the URL with ?modelVersionId=… (or the version's download link)"
    )


def download(url: str, dest_dir: Path, token: str = "") -> Path:
    """Stream to ``<name>.part`` and rename when complete. The Bearer token is dropped by
    requests when a redirect leaves the host (e.g. Civitai -> its storage CDN)."""
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code in (401, 403):
            raise DownloadError(f"access denied ({response.status_code}): this model needs a "
                                "token (Colab Secrets HF_TOKEN / CIVITAI_TOKEN)")  # fmt: skip
        response.raise_for_status()
        if response.headers.get("Content-Type", "").startswith("text/html"):
            raise DownloadError("the link returned a web page, not a file")
        final = Path(dest_dir) / _filename(response)
        total = int(response.headers.get("Content-Length") or 0)
        if final.is_file() and final.stat().st_size == total:
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        done, shown = 0, 0.0
        with open(part, "wb") as fh:
            for chunk in response.iter_content(8 * 2**20):
                fh.write(chunk)
                done += len(chunk)
                if time.monotonic() - shown > 2:  # one updating progress line
                    shown = time.monotonic()
                    print(f"\r{final.name}: {done / 2**30:.2f}/{total / 2**30:.2f} GiB",
                          end="", flush=True)  # fmt: skip
    if total and part.stat().st_size != total:
        raise DownloadError("download incomplete; run the cell again")
    print("\r", end="")
    return part.replace(final)


def _filename(response) -> str:
    """Content-Disposition (or a signed URL's response-content-disposition), else the URL."""
    candidates = [response.headers.get("Content-Disposition", "")]
    candidates += parse_qs(urlparse(response.url).query).get("response-content-disposition", [])
    for disposition in candidates:
        if m := re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition, re.IGNORECASE):
            return Path(unquote(m.group(1))).name
    return Path(unquote(urlparse(response.url).path)).name or "model.safetensors"


def validate_checkpoint(path: Path) -> None:
    """Must be a real, full-size ``.safetensors`` file (not an error page or a LoRA)."""
    if path.suffix.lower() != ".safetensors":
        raise DownloadError(f"{path.name}: only .safetensors checkpoints are supported")
    if path.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise DownloadError(f"{path.name} is too small for a full checkpoint (LoRA/embedding?)")
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        try:
            if length > 100 * 2**20 or not isinstance(json.loads(fh.read(length)), dict):
                raise ValueError
        except ValueError as exc:
            raise DownloadError(f"{path.name} is not a valid safetensors file") from exc


# ---------------------------------------------------------------------------
# Server (SPEC §21.7-21.8)
# ---------------------------------------------------------------------------


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def _exited(pid: int) -> bool:
    try:  # reap our own child: a zombie still "exists" for killpg(pid, 0)
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return True
    except ChildProcessError:
        pass
    try:
        os.killpg(pid, 0)
        return False
    except ProcessLookupError:
        return True


def stop_server(pid_file: Path) -> None:
    """Stop the server started by ``start_server`` (no-op if none is running)."""
    try:
        pid = int(pid_file.read_text())
        os.killpg(pid, signal.SIGTERM)
        if not _wait(lambda: _exited(pid), 15):
            os.killpg(pid, signal.SIGKILL)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass
    pid_file.unlink(missing_ok=True)


def start_server(cmd: list[str], *, cwd: Path, port: int) -> None:
    """(Re)start ``cmd`` in the background; output goes to ``cwd/server.log`` only."""
    pid_file, log_file = cwd / "server.pid", cwd / "server.log"
    stop_server(pid_file)
    if not _wait(lambda: not port_open(port), 10):
        raise RuntimeError(f"port {port} is already in use")
    with open(log_file, "w") as log:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)  # fmt: skip
    pid_file.write_text(str(process.pid))
    started = _wait(lambda: port_open(port) or process.poll() is not None, 60)
    if not started or process.poll() is not None:
        raise RuntimeError(f"server did not start:\n{log_file.read_text()[-3000:]}")


def show_ui(port: int, height: int = 1100) -> None:
    """Embed the UI through Colab's built-in kernel proxy (iframe helper)."""
    try:
        from google.colab import output  # type: ignore[import-not-found]
    except ImportError:
        print(f"Open http://127.0.0.1:{port}/")
        return
    output.serve_kernel_port_as_iframe(port, height=height)
