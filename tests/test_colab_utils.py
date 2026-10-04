"""Tests for notebooks/colab_utils.py against a local HTTP server (no internet)."""

import json
import socket
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("requests")

from notebooks import colab_utils as cu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_safetensors(prefix="cond_stage_model.transformer.", size=4096) -> bytes:
    header = {
        f"{prefix}weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "__metadata__": {"format": "pt"},
    }
    raw = json.dumps(header).encode()
    body = struct.pack("<Q", len(raw)) + raw + b"\x00\x00"
    return body + b"\x01" * (size - len(body))


SD15 = fake_safetensors()
SDXL = fake_safetensors("conditioner.embedders.1.model.")
LORA = fake_safetensors("lora_unet_down_")


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        self.seen.append(
            {"path": self.path, "host": self.headers["Host"], "auth": self.headers["Authorization"]}
        )
        route = self.routes.get(path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        route(self)


def send_file(data: bytes, *, disposition=None, require_auth=False, drop_after=None):
    def handler(h):
        if require_auth and h.headers["Authorization"] != "Bearer secret":
            h.send_response(401)
            h.end_headers()
            return
        start = 0
        if rng := h.headers["Range"]:
            start = int(rng.split("=")[1].split("-")[0])
            h.send_response(206)
            h.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", str(len(data) - start))
        if disposition:
            h.send_header("Content-Disposition", disposition)
        h.end_headers()
        body = data[start:]
        if drop_after is not None and not h.headers["Range"]:
            h.wfile.write(body[:drop_after])
            h.wfile.flush()
            h.close_connection = True
            h.connection.shutdown(socket.SHUT_RDWR)
            return
        h.wfile.write(body)

    return handler


def send_json(obj):
    def handler(h):
        raw = json.dumps(obj).encode()
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    return handler


def redirect(location):
    def handler(h):
        h.send_response(302)
        h.send_header("Location", location)
        h.end_headers()

    return handler


def html_page(h):
    raw = b"<html>login</html>"
    h.send_response(200)
    h.send_header("Content-Type", "text/html; charset=utf-8")
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    h.wfile.write(raw)


@pytest.fixture
def web():
    Handler.routes = {}
    Handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield {
        "routes": Handler.routes,
        "seen": Handler.seen,
        "a": f"http://127.0.0.1:{port}",  # "api" host
        "b": f"http://localhost:{port}",  # "storage" host (different hostname)
    }
    server.shutdown()


QUIET = dict(log=lambda m: None)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("/content/drive/MyDrive/m.safetensors", "local"),
        ("models/m.safetensors", "local"),
        ("https://huggingface.co/a/b/blob/main/m.safetensors", "huggingface"),
        ("https://hf.co/a/b/resolve/main/m.safetensors", "huggingface"),
        ("https://civitai.com/models/1/x", "civitai"),
        ("https://www.civitai.com/api/download/models/2", "civitai"),
        ("https://example.com/m.safetensors", "direct"),
    ],
)
def test_classify_source(source, kind):
    assert cu.classify_source(source) == kind


def test_parse_hf_url():
    t = cu.parse_hf_url(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/"
        "sd_xl_base_1.0.safetensors?download=true"
    )
    assert t == cu.HFTarget(
        "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "main"
    )
    t = cu.parse_hf_url("https://huggingface.co/org/repo/resolve/v2.0/sub/dir/m%20x.safetensors")
    assert (t.revision, t.filename) == ("v2.0", "sub/dir/m x.safetensors")
    with pytest.raises(cu.DownloadError, match="point at a file"):
        cu.parse_hf_url("https://huggingface.co/org/repo")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://civitai.com/api/download/models/12345", cu.CivitaiTarget(None, 12345)),
        (
            "https://civitai.com/api/download/models/12345?type=Model&format=SafeTensor",
            cu.CivitaiTarget(None, 12345),
        ),
        ("https://civitai.com/models/777/cool-model", cu.CivitaiTarget(777, None)),
        ("https://civitai.com/models/777/cool?modelVersionId=99", cu.CivitaiTarget(777, 99)),
    ],
)
def test_parse_civitai_url(url, expected):
    assert cu.parse_civitai_url(url) == expected


def test_parse_civitai_url_rejects_other_pages():
    with pytest.raises(cu.DownloadError):
        cu.parse_civitai_url("https://civitai.com/images/5")


def test_filename_from_response_variants():
    class R:
        def __init__(self, headers=None, url="http://x/y", history=()):
            self.headers, self.url, self.history = headers or {}, url, list(history)

    assert (
        cu.filename_from_response(
            R({"Content-Disposition": 'attachment; filename="a b.safetensors"'})
        )
        == "a b.safetensors"
    )
    utf8 = "attachment; filename*=UTF-8''caf%C3%A9.safetensors"
    assert cu.filename_from_response(R({"Content-Disposition": utf8})) == "café.safetensors"
    signed = "http://cdn/x?response-content-disposition=" + quote(
        'attachment; filename="z.safetensors"'
    )
    assert cu.filename_from_response(R(url=signed)) == "z.safetensors"
    evil = 'attachment; filename="../../etc/passwd.safetensors"'
    assert cu.filename_from_response(R({"Content-Disposition": evil})) == "passwd.safetensors"
    assert cu.filename_from_response(R()) is None


# ---------------------------------------------------------------------------
# download_url
# ---------------------------------------------------------------------------


def test_download_streams_names_and_renames(web, tmp_path):
    web["routes"]["/f"] = send_file(SD15, disposition='attachment; filename="model.safetensors"')
    path = cu.download_url(web["a"] + "/f", tmp_path, **QUIET)
    assert path == tmp_path / "model.safetensors"
    assert path.read_bytes() == SD15
    assert not list(tmp_path.glob("*.part"))


def test_existing_file_is_reused(web, tmp_path):
    (tmp_path / "m.safetensors").write_bytes(SD15)
    path = cu.download_url(web["a"] + "/f", tmp_path, filename="m.safetensors", **QUIET)
    assert path.read_bytes() == SD15 and web["seen"] == []  # no request at all


def test_interrupted_download_resumes_with_range(web, tmp_path):
    web["routes"]["/f"] = send_file(SD15, drop_after=1000)
    with pytest.raises(Exception):  # noqa: B017 — connection drop surfaces from requests
        cu.download_url(
            web["a"] + "/f", tmp_path, filename="m.safetensors", chunk_bytes=256, **QUIET
        )
    part = tmp_path / "m.safetensors.part"
    assert part.is_file() and not (tmp_path / "m.safetensors").exists()
    assert 0 < part.stat().st_size < len(SD15)

    logs = []
    path = cu.download_url(web["a"] + "/f", tmp_path, filename="m.safetensors", log=logs.append)
    assert path.read_bytes() == SD15
    assert any("Resuming" in line for line in logs)
    assert any(req.get("path") == "/f" for req in web["seen"])


def test_sha256_mismatch_deletes_part(web, tmp_path):
    web["routes"]["/f"] = send_file(SD15)
    with pytest.raises(cu.DownloadError, match="SHA256 mismatch"):
        cu.download_url(
            web["a"] + "/f", tmp_path, filename="m.safetensors", expected_sha256="0" * 64, **QUIET
        )
    assert list(tmp_path.iterdir()) == []
    import hashlib

    path = cu.download_url(
        web["a"] + "/f",
        tmp_path,
        filename="m.safetensors",
        expected_sha256=hashlib.sha256(SD15).hexdigest().upper(),
        **QUIET,
    )
    assert path.is_file()


def test_token_sent_to_api_host_but_not_storage_host(web, tmp_path):
    web["routes"]["/api/download"] = redirect(web["b"] + "/storage/file?sig=1")
    web["routes"]["/storage/file"] = send_file(
        SD15, disposition='attachment; filename="c.safetensors"'
    )
    path = cu.download_url(web["a"] + "/api/download", tmp_path, token="secret", **QUIET)
    assert path.name == "c.safetensors"
    api = [r for r in web["seen"] if r["path"].startswith("/api/")]
    storage = [r for r in web["seen"] if r["path"].startswith("/storage/")]
    assert api and all(r["auth"] == "Bearer secret" for r in api)
    assert storage and all(r["auth"] is None for r in storage)


@pytest.mark.parametrize(
    ("route", "match"),
    [
        (html_page, "web page"),
        (send_file(SD15, require_auth=True), "needs a token"),
    ],
)
def test_download_errors_are_clear(web, tmp_path, route, match):
    web["routes"]["/f"] = route
    with pytest.raises(cu.DownloadError, match=match):
        cu.download_url(web["a"] + "/f?token=abc", tmp_path, **QUIET)


def test_missing_file_is_404(web, tmp_path):
    with pytest.raises(cu.DownloadError, match="404"):
        cu.download_url(web["a"] + "/nope", tmp_path, **QUIET)


def test_redact_hides_tokens():
    assert "abc" not in cu._redact("https://x/y?token=abc&X-Amz-Signature=def")


# ---------------------------------------------------------------------------
# Safetensors validation
# ---------------------------------------------------------------------------


def test_read_header_and_guess_family(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(SDXL)
    info = cu.read_safetensors_header(p)
    assert info.num_tensors == 1 and info.metadata == {"format": "pt"}
    assert cu.guess_family(info.keys) == "sdxl"
    assert cu.guess_family(["cond_stage_model.transformer.text_model.x"]) == "sd15"
    assert cu.guess_family(["cond_stage_model.model.transformer.x"]) == "sd2"
    assert cu.guess_family(["lora_unet_x"]) == "lora"
    assert cu.guess_family(["something.else"]) is None


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"<!DOCTYPE html><html>", "HTML"),
        (b'{"error": "unauthorized"}', "HTML/JSON"),
        (struct.pack("<Q", 10**12) + b"x" * 100, "not a valid"),
        (struct.pack("<Q", 20) + b"not json at all!!!!!", "corrupt"),
        (b"abc", "too small"),
    ],
)
def test_bad_safetensors_rejected(tmp_path, content, match):
    p = tmp_path / "m.safetensors"
    p.write_bytes(content)
    with pytest.raises(cu.DownloadError, match=match):
        cu.read_safetensors_header(p)


def test_validate_checkpoint(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(SD15)
    logs = []
    cu.validate_checkpoint(p, "sd15", min_bytes=100, log=logs.append)
    assert logs[-1].startswith("Checkpoint OK")

    logs.clear()
    cu.validate_checkpoint(p, "sdxl", min_bytes=100, log=logs.append)
    assert any("looks like sd15" in line for line in logs)  # warning only

    with pytest.raises(cu.DownloadError, match="too small"):
        cu.validate_checkpoint(p, "sd15", log=logs.append)  # default 500 MiB minimum
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(SD15)
    with pytest.raises(cu.DownloadError, match=".safetensors"):
        cu.validate_checkpoint(ckpt, "sd15", min_bytes=100)
    lora = tmp_path / "l.safetensors"
    lora.write_bytes(LORA)
    logs.clear()
    cu.validate_checkpoint(lora, "sd15", min_bytes=100, log=logs.append)
    assert any("looks like a lora" in line for line in logs)


# ---------------------------------------------------------------------------
# Civitai + fetch_checkpoint
# ---------------------------------------------------------------------------


def civitai_version(web, *, base="SDXL 1.0", files=None, sha=None):
    import hashlib

    return {
        "id": 123,
        "name": "v2",
        "baseModel": base,
        "model": {"name": "Cool XL", "type": "Checkpoint"},
        "files": files
        if files is not None
        else [
            {
                "name": "cool_vae.safetensors",
                "type": "VAE",
                "metadata": {"format": "SafeTensor"},
                "downloadUrl": web["a"] + "/api/download/models/123?type=VAE",
            },
            {
                "name": "cool.safetensors",
                "type": "Model",
                "primary": True,
                "sizeKB": len(SDXL) / 1024,
                "metadata": {"format": "SafeTensor"},
                "hashes": {"SHA256": sha or hashlib.sha256(SDXL).hexdigest().upper()},
                "downloadUrl": web["a"] + "/api/download/models/123",
            },
        ],
    }


def test_fetch_civitai_end_to_end(web, tmp_path):
    web["routes"]["/api/v1/model-versions/123"] = send_json(civitai_version(web))
    web["routes"]["/api/download/models/123"] = redirect(web["b"] + "/cdn/obj?sig=1")
    web["routes"]["/cdn/obj"] = send_file(SDXL)
    logs = []
    path = cu.fetch_checkpoint(
        "https://civitai.com/models/9/cool?modelVersionId=123",
        "sdxl",
        tmp_path,
        civitai_token="secret",
        civitai_api=web["a"] + "/api/v1",
        min_bytes=100,
        log=logs.append,
    )
    assert path == tmp_path / "cool.safetensors" and path.read_bytes() == SDXL
    assert any("Civitai: Cool XL · v2 · SDXL 1.0" in line for line in logs)
    assert not any("WARNING" in line for line in logs)
    cdn = [r for r in web["seen"] if r["path"].startswith("/cdn/")]
    assert cdn and all(r["auth"] is None for r in cdn)


def test_fetch_civitai_latest_version_and_family_warning(web, tmp_path):
    web["routes"]["/api/v1/models/9"] = send_json(
        {
            "name": "Cool",
            "type": "Checkpoint",
            "modelVersions": [civitai_version(web, base="SD 1.5")],
        }
    )
    web["routes"]["/api/download/models/123"] = send_file(SDXL)
    logs = []
    cu.fetch_checkpoint(
        "https://civitai.com/models/9/cool",
        "sdxl",
        tmp_path,
        civitai_api=web["a"] + "/api/v1",
        min_bytes=100,
        log=logs.append,
    )
    assert any("base model is 'SD 1.5'" in line for line in logs)


def test_civitai_version_without_safetensors(web):
    version = civitai_version(
        web, files=[{"name": "m.ckpt", "type": "Model", "metadata": {"format": "PickleTensor"}}]
    )
    with pytest.raises(cu.DownloadError, match="no .safetensors"):
        cu.pick_civitai_file(version)


@pytest.mark.parametrize(
    ("base", "family", "warn"),
    [("SDXL 1.0", "sdxl", False), ("Pony", "sdxl", False), ("Illustrious", "sdxl", False),
     ("SD 1.5", "sd15", False), ("SD 1.5", "sdxl", True), ("Flux.1 D", "sdxl", True)],
)  # fmt: skip
def test_civitai_family_warning(base, family, warn):
    info = cu.CivitaiFile("u", "f", None, None, base_model=base, model_type="Checkpoint")
    assert (cu.civitai_family_warning(info, family) is not None) is warn
    lora = cu.CivitaiFile("u", "f", None, None, base_model="SDXL 1.0", model_type="LORA")
    assert "LORA" in cu.civitai_family_warning(lora, "sdxl")


def test_fetch_huggingface_uses_hub(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(SD15)
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    path = cu.fetch_checkpoint(
        "https://huggingface.co/org/repo/blob/main/m.safetensors",
        "sd15",
        tmp_path,
        hf_token="hf_x",
        min_bytes=100,
        **QUIET,
    )
    assert path == tmp_path / "m.safetensors"
    assert calls["repo_id"] == "org/repo" and calls["token"] == "hf_x"
    assert calls["revision"] == "main" and calls["local_dir"] == tmp_path


def test_fetch_rejects_non_safetensors_hf_and_bad_input(tmp_path):
    with pytest.raises(cu.DownloadError, match=".safetensors"):
        cu.fetch_checkpoint(
            "https://huggingface.co/o/r/blob/main/m.ckpt", "sd15", tmp_path, **QUIET
        )
    with pytest.raises(cu.DownloadError, match="empty"):
        cu.fetch_checkpoint("  ", "sd15", tmp_path, **QUIET)
    with pytest.raises(cu.DownloadError, match="MODEL_FAMILY"):
        cu.fetch_checkpoint("x.safetensors", "auto", tmp_path, **QUIET)
    with pytest.raises(cu.DownloadError, match="not found"):
        cu.fetch_checkpoint(str(tmp_path / "missing.safetensors"), "sd15", tmp_path, **QUIET)


def test_fetch_local_path(tmp_path):
    p = tmp_path / "drive" / "m.safetensors"
    p.parent.mkdir()
    p.write_bytes(SD15)
    assert cu.fetch_checkpoint(str(p), "sd15", tmp_path, min_bytes=100, **QUIET) == p


def test_fetch_direct_url(web, tmp_path):
    web["routes"]["/m.safetensors"] = send_file(SD15)
    path = cu.fetch_checkpoint(
        web["a"] + "/m.safetensors", "sd15", tmp_path, min_bytes=100, **QUIET
    )
    assert path.name == "m.safetensors"


# ---------------------------------------------------------------------------
# Secrets, preflight, server lifecycle
# ---------------------------------------------------------------------------


def test_get_secret_fallbacks(monkeypatch):
    monkeypatch.delenv("SDMT_TEST_TOKEN", raising=False)
    assert cu.get_secret("SDMT_TEST_TOKEN") == ""
    assert cu.get_secret("SDMT_TEST_TOKEN", " typed ") == "typed"
    monkeypatch.setenv("SDMT_TEST_TOKEN", "from-env")
    assert cu.get_secret("SDMT_TEST_TOKEN") == "from-env"


def test_runtime_preflight_is_silent_and_summarises(capsys):
    info = cu.runtime_preflight()
    assert capsys.readouterr().out == ""
    assert info["summary"].startswith(f"Python {info['python']} · ")
    assert isinstance(info["warnings"], list) and info["cuda"] in (True, False)
    if not info["cuda"]:
        assert info["summary"].endswith("no GPU")


def test_live_line_prints_outside_ipython(capsys):
    line = cu.LiveLine()
    line("downloading 1%")
    line("WARNING: family mismatch")
    out = capsys.readouterr().out
    assert "downloading 1%" in out and "WARNING: family mismatch" in out


def test_install_requirements_is_silent_and_guards_torch(monkeypatch, tmp_path):
    calls, logs = [], []
    monkeypatch.setattr(cu, "run", lambda cmd, what, cwd=None: calls.append(cmd) or "")
    versions = iter(["2.11.0", "2.11.0"])
    monkeypatch.setattr(cu, "torch_version", lambda: next(versions))
    assert cu.install_requirements(tmp_path, log=logs.append) == "2.11.0"
    assert logs == [] and "-r" in calls[0]
    versions = iter(["2.11.0", "2.12.0"])
    cu.install_requirements(tmp_path, log=logs.append)
    assert "pip changed torch 2.11.0 → 2.12.0" in logs[0]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api_status(port):
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as r:
        return json.loads(r.read())


def test_server_lifecycle_with_mock(tmp_path):
    import time

    port = free_port()
    log_file, pid_file = tmp_path / "server.log", tmp_path / "server.pid"
    cmd = [
        sys.executable, "-m", "src.cli", "serve", "--mock", "--model-family", "sd15",
        "--port", str(port), "--mock-load-seconds", "0.3",
        "--inputs-dir", str(tmp_path / "in"), "--outputs-dir", str(tmp_path / "out"),
    ]  # fmt: skip
    logs = []
    try:
        cu.start_server(
            cmd, cwd=REPO_ROOT, log_file=log_file, pid_file=pid_file, port=port, log=logs.append
        )
        assert cu.port_open(port) and pid_file.is_file()
        deadline = time.monotonic() + 30
        while api_status(port)["backend_state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert api_status(port)["backend_state"] == "ready"
        assert "Backend: ready" in log_file.read_text()  # server logs go to the file only

        # restarting stops the old process first
        cu.start_server(
            cmd, cwd=REPO_ROOT, log_file=log_file, pid_file=pid_file, port=port, log=logs.append
        )
        assert cu.port_open(port)
    finally:
        assert cu.stop_server(pid_file, port, log=logs.append)
    assert not cu.port_open(port) and not pid_file.exists()
    assert cu.stop_server(pid_file, port) is False  # idempotent


def test_server_start_failure_shows_log(tmp_path):
    with pytest.raises(RuntimeError, match="did not start") as info:
        cu.start_server(
            [sys.executable, "-c", "print('boom: bad config'); raise SystemExit(3)"],
            cwd=tmp_path,
            log_file=tmp_path / "s.log",
            pid_file=tmp_path / "s.pid",
            port=free_port(),
            timeout=10,
            **QUIET,
        )
    assert "boom: bad config" in str(info.value)


def test_show_ui_outside_colab():
    logs = []
    cu.show_ui(8000, log=logs.append)
    assert "127.0.0.1:8000" in logs[0]
