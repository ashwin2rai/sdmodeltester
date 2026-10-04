"""Colab notebook utilities: runtime preflight, checkpoint download, server launch.

These are notebook operations, deliberately kept out of ``src/`` (SPEC §21.1): the
notebook clones this repo and imports this module so its cells stay short. Nothing here
implements image generation.

Download design (improving on the A1111 fast-stable-diffusion notebook's helpers):
- Hugging Face URLs go through ``huggingface_hub.hf_hub_download`` (token, resume, xet).
- Civitai URLs (model page, ``?modelVersionId=``, or ``/api/download/models/{id}``) are
  resolved through the public API to the version's primary SafeTensor file, its
  filename, size and SHA256.
- Everything else is streamed with ``requests`` to ``<name>.part`` and renamed only when
  complete, so a partial file never looks finished; an interrupted ``.part`` is resumed
  with an HTTP Range request.
- Tokens go only in an ``Authorization`` header, which ``requests`` drops when a redirect
  leaves the original host (Civitai -> storage CDN), and are never printed.
- The result must be a real ``.safetensors`` file (header parsed), big enough to be a
  full checkpoint, and is checked against the chosen model family (warning only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

CHUNK_BYTES = 8 * 2**20
MIN_CHECKPOINT_BYTES = 500 * 2**20  # LoRAs/embeddings are smaller than any full SD checkpoint
MAX_HEADER_BYTES = 100 * 2**20
CIVITAI_API = "https://civitai.com/api/v1"
USER_AGENT = "sdmodeltester-colab/1.0"

Printer = Callable[[str], None]


class DownloadError(RuntimeError):
    """A checkpoint could not be downloaded or failed validation."""


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def get_secret(name: str, fallback: str = "") -> str:
    """Colab Secrets (key icon in the sidebar) first, then the form fallback, then env."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get(name)
        if value:
            return value.strip()
    except Exception:  # noqa: BLE001 — not in Colab, secret missing, or access not granted
        pass
    return (fallback or os.environ.get(name, "")).strip()


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HFTarget:
    repo_id: str
    filename: str
    revision: str = "main"
    repo_type: str = "model"


@dataclass(frozen=True)
class CivitaiTarget:
    model_id: int | None = None
    version_id: int | None = None


def classify_source(source: str) -> str:
    """'local' | 'huggingface' | 'civitai' | 'direct'."""
    source = source.strip()
    if not re.match(r"^https?://", source, re.IGNORECASE):
        return "local"
    host = (urlparse(source).hostname or "").lower()
    if host in ("huggingface.co", "www.huggingface.co", "hf.co"):
        return "huggingface"
    if host == "civitai.com" or host.endswith(".civitai.com") or "civitai" in host.split("."):
        return "civitai"
    return "direct"


def parse_hf_url(url: str) -> HFTarget:
    """``https://huggingface.co/{repo}/(blob|resolve)/{revision}/{path}`` -> HFTarget."""
    parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
    repo_type = "model"
    if parts and parts[0] in ("datasets", "spaces"):
        repo_type = parts[0][:-1]
        parts = parts[1:]
    if len(parts) < 5 or parts[2] not in ("blob", "resolve"):
        raise DownloadError(
            "Hugging Face link must point at a file, e.g. "
            "https://huggingface.co/<org>/<repo>/blob/main/<file>.safetensors"
        )
    return HFTarget(
        repo_id=f"{parts[0]}/{parts[1]}",
        revision=parts[3],
        filename="/".join(parts[4:]),
        repo_type=repo_type,
    )


def parse_civitai_url(url: str) -> CivitaiTarget:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    path = parsed.path
    if m := re.search(r"/api/download/models/(\d+)", path):
        return CivitaiTarget(version_id=int(m.group(1)))
    if m := re.search(r"/api/v1/model-versions/(\d+)", path):
        return CivitaiTarget(version_id=int(m.group(1)))
    if m := re.search(r"/models/(\d+)", path):
        version = query.get("modelVersionId", [None])[0]
        return CivitaiTarget(model_id=int(m.group(1)), version_id=int(version) if version else None)
    raise DownloadError(
        "Civitai link not recognised; use the model page URL "
        "(https://civitai.com/models/<id>?modelVersionId=<id>) or the download link "
        "(https://civitai.com/api/download/models/<versionId>)"
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _session(token: str = ""):
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if token:
        # requests strips Authorization when a redirect changes host, so the token is
        # sent to Civitai/HF but not to the signed storage URL they redirect to.
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def filename_from_response(response: Any) -> str | None:
    """Filename from Content-Disposition, or from a signed URL's
    ``response-content-disposition`` query parameter (how Civitai's CDN names files)."""
    candidates = [response.headers.get("Content-Disposition", "")]
    for r in [*getattr(response, "history", []), response]:
        query = parse_qs(urlparse(r.url).query)
        candidates += query.get("response-content-disposition", [])
    for disposition in candidates:
        if not disposition:
            continue
        if m := re.search(r"filename\*\s*=\s*[^']*''([^;]+)", disposition, re.IGNORECASE):
            return safe_filename(unquote(m.group(1)))
        if m := re.search(r'filename\s*=\s*"?([^";]+)"?', disposition, re.IGNORECASE):
            return safe_filename(unquote(m.group(1)))
    return None


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip().strip(".")
    name = re.sub(r"[^\w.\-+() ]", "_", name)
    if not name:
        raise DownloadError("could not determine a safe filename")
    return name


def _format_bytes(n: float) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.1f} MiB"


class _Progress:
    def __init__(self, total: int | None, start: int, log: Printer, interval: float):
        self.total, self.done, self.log, self.interval = total, start, log, interval
        self.started = self.last = time.monotonic()
        self.start = start

    def update(self, n: int) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            self.report()

    def report(self) -> None:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = (self.done - self.start) / elapsed
        pct = f" ({self.done / self.total:.0%})" if self.total else ""
        total = f" / {_format_bytes(self.total)}" if self.total else ""
        self.log(f"  {_format_bytes(self.done)}{total}{pct} · {rate / 2**20:.1f} MiB/s")


def download_url(
    url: str,
    dest_dir: Path,
    *,
    token: str = "",
    filename: str | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    log: Printer = print,
    progress_interval: float = 10.0,
    chunk_bytes: int = CHUNK_BYTES,
) -> Path:
    """Stream ``url`` into ``dest_dir`` via ``<name>.part`` (resumable), then rename.

    If ``filename`` is not given it is taken from the response headers. An existing
    complete file (matching ``expected_size`` when known) is reused without downloading.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        final = dest_dir / safe_filename(filename)
        if final.is_file() and (expected_size is None or final.stat().st_size == expected_size):
            log(f"Already downloaded: {final}")
            return final

    session = _session(token)
    with session.get(url, stream=True, allow_redirects=True, timeout=60) as response:
        _raise_for_status(response, url)
        name = filename or filename_from_response(response) or _url_basename(response.url)
        final = dest_dir / safe_filename(name)
        total = expected_size or _content_length(response)
        if final.is_file() and (total is None or final.stat().st_size == total):
            log(f"Already downloaded: {final}")
            return final
        part = final.with_name(final.name + ".part")
        resume_from = part.stat().st_size if part.is_file() else 0
        if not resume_from:  # fresh download: stream this response directly
            _write_stream(response, part, "wb", 0, total, log, progress_interval, chunk_bytes)

    if resume_from and not (total and resume_from >= total):
        _stream_to_part(session, url, part, resume_from, total, log, progress_interval, chunk_bytes)

    size = part.stat().st_size
    if total is not None and size != total:
        raise DownloadError(f"download incomplete: {size} of {total} bytes ({part})")
    if expected_sha256:
        log("Verifying SHA256…")
        digest = sha256_file(part)
        if digest.lower() != expected_sha256.lower():
            part.unlink()
            raise DownloadError(
                f"SHA256 mismatch (got {digest[:12]}…, expected {expected_sha256[:12]}…)"
            )
    part.replace(final)
    log(f"Downloaded {final.name} ({_format_bytes(size)})")
    return final


def _stream_to_part(
    session: Any,
    url: str,
    part: Path,
    resume_from: int,
    total: int | None,
    log: Printer,
    interval: float,
    chunk_bytes: int,
) -> None:
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with session.get(url, stream=True, headers=headers, timeout=60) as response:
        if response.status_code == 206:
            mode, start = "ab", resume_from
            log(f"Resuming download at {_format_bytes(resume_from)}")
        elif response.status_code == 416:
            return  # nothing left to fetch; the size check decides
        else:  # server ignored Range: start over
            _raise_for_status(response, url)
            mode, start = "wb", 0
        _write_stream(response, part, mode, start, total, log, interval, chunk_bytes)


def _write_stream(
    response: Any,
    part: Path,
    mode: str,
    start: int,
    total: int | None,
    log: Printer,
    interval: float,
    chunk_bytes: int,
) -> None:
    progress = _Progress(total, start, log, interval)
    with open(part, mode) as fh:
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            if chunk:
                fh.write(chunk)
                progress.update(len(chunk))
    progress.report()


def _content_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def _url_basename(url: str) -> str:
    return unquote(Path(urlparse(url).path).name) or "model.safetensors"


def _raise_for_status(response: Any, url: str) -> None:
    if response.status_code in (401, 403):
        raise DownloadError(
            f"access denied ({response.status_code}) for {_redact(url)}: this model needs a "
            "token (add HF_TOKEN / CIVITAI_TOKEN in Colab Secrets) or you must accept its "
            "license on the website first"
        )
    if response.status_code == 404:
        raise DownloadError(f"not found (404): {_redact(url)}")
    if response.status_code >= 400:
        raise DownloadError(f"HTTP {response.status_code} for {_redact(url)}")
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("text/html"):
        raise DownloadError(
            f"{_redact(url)} returned a web page, not a file — use the file's download link"
        )


def _redact(url: str) -> str:
    return re.sub(r"(token|signature|X-Amz-[A-Za-z]+)=[^&]+", r"\1=…", url)


def sha256_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Civitai
# ---------------------------------------------------------------------------


@dataclass
class CivitaiFile:
    download_url: str
    filename: str
    size_bytes: int | None
    sha256: str | None
    model_name: str = ""
    version_name: str = ""
    model_type: str = ""
    base_model: str = ""


def _get_json(session: Any, url: str) -> Any:
    response = session.get(url, timeout=60)
    _raise_for_status(response, url)
    return response.json()


def pick_civitai_file(version: dict, model: dict | None = None) -> CivitaiFile:
    """Choose the version's SafeTensor checkpoint file (primary first)."""
    files = version.get("files") or []
    safetensors = [
        f
        for f in files
        if (f.get("metadata") or {}).get("format") == "SafeTensor"
        or str(f.get("name", "")).lower().endswith(".safetensors")
    ]
    if not safetensors:
        names = ", ".join(str(f.get("name")) for f in files) or "none"
        raise DownloadError(f"this Civitai version has no .safetensors file (files: {names})")
    model_files = [f for f in safetensors if f.get("type", "Model") == "Model"] or safetensors
    chosen = next((f for f in model_files if f.get("primary")), model_files[0])
    size_kb = chosen.get("sizeKB")
    model = model or version.get("model") or {}
    return CivitaiFile(
        download_url=chosen.get("downloadUrl")
        or f"https://civitai.com/api/download/models/{version['id']}",
        filename=chosen["name"],
        size_bytes=int(round(size_kb * 1024)) if size_kb else None,
        sha256=(chosen.get("hashes") or {}).get("SHA256"),
        model_name=model.get("name", ""),
        version_name=version.get("name", ""),
        model_type=model.get("type", ""),
        base_model=version.get("baseModel", ""),
    )


def resolve_civitai(url: str, token: str = "", api: str = CIVITAI_API) -> CivitaiFile:
    target = parse_civitai_url(url)
    session = _session(token)
    model = None
    if target.version_id is None:
        model = _get_json(session, f"{api}/models/{target.model_id}")
        versions = model.get("modelVersions") or []
        if not versions:
            raise DownloadError("this Civitai model has no versions")
        version = versions[0]  # newest first
    else:
        version = _get_json(session, f"{api}/model-versions/{target.version_id}")
    return pick_civitai_file(version, model)


CIVITAI_FAMILY = {
    "sd15": ("SD 1.5", "SD 1.4", "SD 1.5 LCM", "SD 1.5 Hyper"),
    "sdxl": ("SDXL", "Pony", "Illustrious", "NoobAI"),
}


def civitai_family_warning(info: CivitaiFile, family: str) -> str | None:
    """Warn-only hint when Civitai metadata disagrees with the chosen family."""
    if info.model_type and info.model_type.lower() != "checkpoint":
        return f"Civitai says this is a {info.model_type}, not a full checkpoint"
    base = info.base_model
    if not base:
        return None
    if any(base.startswith(prefix) for prefix in CIVITAI_FAMILY[family]):
        return None
    return f"Civitai base model is '{base}', but MODEL_FAMILY is '{family}'"


# ---------------------------------------------------------------------------
# Safetensors validation
# ---------------------------------------------------------------------------


@dataclass
class SafetensorsInfo:
    num_tensors: int
    keys: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def read_safetensors_header(path: Path) -> SafetensorsInfo:
    """Parse the header (8-byte little-endian length + JSON) without loading tensors."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            raise DownloadError(f"{path.name} is too small to be a safetensors file")
        if head.lstrip().startswith((b"<", b"{")):
            raise DownloadError(f"{path.name} looks like an HTML/JSON page, not a checkpoint")
        (length,) = struct.unpack("<Q", head)
        if length <= 1 or length > MAX_HEADER_BYTES:
            raise DownloadError(f"{path.name} is not a valid safetensors file")
        try:
            header = json.loads(fh.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DownloadError(f"{path.name} has a corrupt safetensors header") from exc
    if not isinstance(header, dict):
        raise DownloadError(f"{path.name} has a corrupt safetensors header")
    metadata = header.pop("__metadata__", {}) or {}
    return SafetensorsInfo(len(header), list(header), metadata)


def guess_family(keys: Iterable[str]) -> str | None:
    """Architecture hint from tensor names: 'sd15' | 'sdxl' | 'sd2' | 'diffusers' | other."""
    keys = list(keys)

    def has(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    if has("conditioner.embedders.1.") or has("conditioner.embedders.0.transformer."):
        return "sdxl"
    if has("cond_stage_model.transformer."):
        return "sd15"
    if has("cond_stage_model.model."):
        return "sd2"
    if has("double_blocks.") or has("joint_blocks."):
        return "flux/sd3"
    if has("down_blocks.") or has("unet.down_blocks."):
        return "diffusers"
    if has("lora_unet_") or has("lora_te"):
        return "lora"
    return None


def validate_checkpoint(
    path: Path, family: str, *, min_bytes: int = MIN_CHECKPOINT_BYTES, log: Printer = print
) -> SafetensorsInfo:
    """Raise on anything that cannot be a single-file SD checkpoint; warn on family doubts."""
    path = Path(path)
    if not path.is_file():
        raise DownloadError(f"checkpoint not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise DownloadError(f"{path.name}: V1 only accepts .safetensors checkpoints")
    size = path.stat().st_size
    if size < min_bytes:
        raise DownloadError(
            f"{path.name} is only {_format_bytes(size)} — too small for a full SD checkpoint "
            "(is it a LoRA or embedding?)"
        )
    info = read_safetensors_header(path)
    guessed = guess_family(info.keys)
    if guessed in ("lora", "diffusers", "sd2", "flux/sd3"):
        log(f"WARNING: {path.name} looks like a {guessed} file, not an SD1.5/SDXL checkpoint")
    elif guessed and guessed != family:
        log(
            f"WARNING: {path.name} looks like {guessed}, but MODEL_FAMILY is '{family}'. "
            "Loading will likely fail; change MODEL_FAMILY if so."
        )
    log(f"Checkpoint OK: {path.name} · {_format_bytes(size)} · {info.num_tensors} tensors")
    return info


# ---------------------------------------------------------------------------
# High-level fetch
# ---------------------------------------------------------------------------


def fetch_checkpoint(
    source: str,
    family: str,
    models_dir: Path,
    *,
    hf_token: str = "",
    civitai_token: str = "",
    log: Printer = print,
    min_bytes: int = MIN_CHECKPOINT_BYTES,
    civitai_api: str = CIVITAI_API,
) -> Path:
    """Local path, Hugging Face URL, Civitai URL, or direct URL -> validated checkpoint."""
    source = source.strip()
    if not source:
        raise DownloadError("MODEL_SOURCE is empty: paste a Hugging Face or Civitai link")
    if family not in CIVITAI_FAMILY:
        raise DownloadError(f"MODEL_FAMILY must be 'sd15' or 'sdxl' (got {family!r})")
    models_dir = Path(models_dir)
    kind = classify_source(source)
    log(f"Source: {kind}")

    if kind == "local":
        path = Path(source).expanduser()
    elif kind == "huggingface":
        target = parse_hf_url(source)
        if not target.filename.lower().endswith(".safetensors"):
            raise DownloadError(f"{target.filename}: V1 only accepts .safetensors checkpoints")
        from huggingface_hub import hf_hub_download

        log(f"Downloading {target.filename} from {target.repo_id}@{target.revision}…")
        path = Path(
            hf_hub_download(
                repo_id=target.repo_id,
                filename=target.filename,
                revision=target.revision,
                repo_type=target.repo_type,
                token=hf_token or None,
                local_dir=models_dir,
            )
        )
    elif kind == "civitai":
        info = resolve_civitai(source, civitai_token, api=civitai_api)
        log(
            f"Civitai: {info.model_name} · {info.version_name} · {info.base_model or '?'} "
            f"· {info.filename}"
        )
        if warning := civitai_family_warning(info, family):
            log(f"WARNING: {warning}")
        path = download_url(
            info.download_url,
            models_dir,
            token=civitai_token,
            filename=info.filename,
            expected_size=None,  # sizeKB is rounded; rely on SHA256 instead
            expected_sha256=info.sha256,
            log=log,
        )
    else:
        path = download_url(source, models_dir, token=hf_token, log=log)

    validate_checkpoint(path, family, min_bytes=min_bytes, log=log)
    return path


# ---------------------------------------------------------------------------
# Runtime preflight / installation
# ---------------------------------------------------------------------------


def runtime_preflight() -> dict[str, Any]:
    """Inspect Colab's own Python/torch/GPU (never installs or upgrades anything).

    Returns ``python, torch, cuda, gpu, vram_gib`` plus a one-line ``summary`` and a list
    of ``warnings`` for the notebook to print.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda": False,
        "gpu": None,
        "vram_gib": None,
        "warnings": [],
    }
    try:
        import torch
    except ImportError:
        info["warnings"].append("PyTorch is not installed; this expects Colab's preinstalled torch")
    else:
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu"], info["vram_gib"] = props.name, round(props.total_memory / 2**30, 1)
            if "L4" not in props.name:
                info["warnings"].append(
                    "tuned for an NVIDIA L4; other GPUs may need smaller batches or sizes"
                )
    gpu = f"{info['gpu']} ({info['vram_gib']} GiB)" if info["gpu"] else "no GPU"
    torch_text = f"torch {info['torch']}" if info["torch"] else "no torch"
    info["summary"] = f"Python {info['python']} · {torch_text} · {gpu}"
    return info


def run(cmd: Sequence[str], what: str, *, cwd: Path | None = None) -> str:
    """Quiet on success, full output on failure."""
    result = subprocess.run(list(cmd), capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"{what} failed:\n$ {' '.join(map(str, cmd))}")
        print((result.stdout or "")[-4000:])
        print((result.stderr or "")[-4000:])
        raise RuntimeError(f"{what} failed")
    return result.stdout


def torch_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("torch")
    except Exception:  # noqa: BLE001
        return None


def install_requirements(repo_dir: Path, log: Printer = print) -> str | None:
    """pip-install app + inference deps on top of Colab's stack; never replace torch.

    Silent on success (returns the torch version); warns if pip changed torch anyway.
    """
    before = torch_version()
    run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "-r", str(repo_dir / "requirements.txt"),
            "-r", str(repo_dir / "requirements-inference.txt"),
        ],
        "pip install",
    )  # fmt: skip
    after = torch_version()
    if before != after:
        log(f"WARNING: pip changed torch {before} → {after}. Restart the runtime "
            "(Runtime → Restart session) and report this; the notebook must not replace "
            "Colab's torch.")  # fmt: skip
    return after


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def stop_server(pid_file: Path, port: int | None = None, log: Printer = print) -> bool:
    """Stop a server started by ``start_server`` (safe to call when none is running)."""
    pid_file = Path(pid_file)
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.killpg(pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return False
    if not _wait(lambda: _exited(pid), 15):
        os.killpg(pid, signal.SIGKILL)
        _wait(lambda: _exited(pid), 5)
    pid_file.unlink(missing_ok=True)
    if port is not None:
        _wait(lambda: not port_open(port), 10)
    log(f"Stopped server (pid {pid})")
    return True


def _exited(pid: int) -> bool:
    """True once ``pid`` is gone. Reaps it if it is our child (a zombie still 'exists')."""
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return True
    except ChildProcessError:
        pass  # started by another process (e.g. an earlier kernel); fall through
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def start_server(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    pid_file: Path,
    port: int,
    env: dict[str, str] | None = None,
    timeout: float = 60,
    log: Printer = print,
) -> subprocess.Popen:
    """Start ``cmd`` in the background (own process group), wait until ``port`` answers."""
    stop_server(pid_file, port, log=log)
    if port_open(port):
        raise RuntimeError(f"port {port} is already in use by another process")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    out = open(log_file, "w")  # noqa: SIM115 — owned by the child process
    process = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        stdout=out,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
        start_new_session=True,
    )
    pid_file.write_text(str(process.pid))
    ok = _wait(lambda: port_open(port) or process.poll() is not None, timeout)
    if process.poll() is not None or not ok:
        tail = log_file.read_text()[-4000:] if log_file.exists() else ""
        raise RuntimeError(f"server did not start (exit code {process.poll()}):\n{tail}")
    log(f"Server running (pid {process.pid}) on port {port}; log: {log_file}")
    return process


def _wait(predicate: Callable[[], bool], timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def show_ui(port: int, height: int = 1100, log: Printer = print) -> None:
    """Embed the UI with Colab's built-in kernel proxy (iframe helper; SPEC §21.8)."""
    try:
        from google.colab import output  # type: ignore[import-not-found]
    except ImportError:
        log(f"Not running in Colab; open http://127.0.0.1:{port}/ in a browser.")
        return
    output.serve_kernel_port_as_iframe(port, height=height)
    try:  # a full-tab link as well, when this Colab version can produce one
        from google.colab.output import eval_js  # type: ignore[import-not-found]

        url = eval_js(f"google.colab.kernel.proxyPort({port})")
        log(f"Open in a new tab (only works while this notebook is open): {url}")
    except Exception:  # noqa: BLE001 — iframe is the supported baseline
        pass


def zip_outputs(outputs_dir: Path, archive_base: Path, log: Printer = print) -> Path | None:
    """Zip ``outputs/`` and offer it as a browser download when running in Colab."""
    import shutil

    outputs_dir = Path(outputs_dir)
    if not any(p.suffix == ".png" for p in outputs_dir.rglob("*")):
        log("No images in outputs/ yet.")
        return None
    archive = Path(shutil.make_archive(str(archive_base), "zip", outputs_dir))
    log(f"Created {archive} ({_format_bytes(archive.stat().st_size)})")
    try:
        from google.colab import files  # type: ignore[import-not-found]

        files.download(str(archive))
    except ImportError:
        pass
    return archive


# ---------------------------------------------------------------------------
# Quiet notebook output
# ---------------------------------------------------------------------------


class LiveLine:
    """A printer that keeps a single, updating status line in Colab/Jupyter.

    Lines starting with WARNING/ERROR are printed permanently; everything else replaces
    the previous status. Outside IPython it simply prints.
    """

    def __init__(self) -> None:
        self._handle = None
        try:
            from IPython import get_ipython
            from IPython.display import Pretty, display

            if get_ipython() is not None:
                self._pretty = Pretty
                self._handle = display(Pretty(""), display_id=True)
        except ImportError:
            pass

    def __call__(self, message: str) -> None:
        if self._handle is None or message.startswith(("WARNING", "ERROR")):
            print(message)
        else:
            self._handle.update(self._pretty(message))


# ---------------------------------------------------------------------------
# Benchmark (Phase 8) from the notebook
# ---------------------------------------------------------------------------


def run_benchmark(
    repo_dir: Path,
    family: str,
    *,
    model_path: Path | None,
    profiles: str = "baseline,compile",
    functional: bool = True,
    mock: bool = False,
) -> dict[str, Any]:
    """Run ``src.cli benchmark --quiet`` and return its JSON summary.

    The CLI prints only the report path; the JSON next to it names the contact sheet.
    """
    cmd = [
        sys.executable, "-m", "src.cli", "benchmark", "--quiet",
        "--model-family", family, "--profiles", profiles,
    ]  # fmt: skip
    cmd += ["--mock"] if mock else ["--model", str(model_path)]
    if not functional:
        cmd.append("--no-functional")
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    report = Path(repo_dir) / lines[-1] if lines else None
    if report is None or not report.with_suffix(".json").is_file():
        print((result.stderr or result.stdout)[-4000:])
        raise RuntimeError(f"benchmark failed (exit code {result.returncode})")
    summary = json.loads(report.with_suffix(".json").read_text())
    summary["exit_code"] = result.returncode
    summary["report"] = str(report)
    if summary.get("contact_sheet"):
        summary["contact_sheet"] = str(Path(repo_dir) / summary["contact_sheet"])
    return summary


def show_benchmark(summary: dict[str, Any]) -> None:
    """Render the benchmark report (and contact sheet) in the notebook."""
    report = Path(summary["report"])
    sheet = summary.get("contact_sheet")
    try:
        from IPython.display import Image, Markdown, display

        display(Markdown(report.read_text()))
        if sheet and Path(sheet).is_file():
            display(Image(filename=sheet))
    except ImportError:
        print(report.read_text())
