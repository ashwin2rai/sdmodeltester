"""Tests for notebooks/colab_utils.py against a local HTTP server (no internet)."""

import json
import socket
import struct
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("requests")

from notebooks import colab_utils as cu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_checkpoint(size=4096) -> bytes:
    header = json.dumps({"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    body = struct.pack("<Q", len(header)) + header
    return body + b"\x00" * (size - len(body))


CKPT = fake_checkpoint()


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.seen.append((self.path, self.headers["Authorization"]))
        route = self.routes.get(self.path.split("?")[0])
        if route is None:
            self.send_response(404)
            self.end_headers()
        else:
            route(self)


def send(data: bytes, content_type="application/octet-stream", disposition=None, auth=False):
    def handler(h):
        if auth and "token=secret" not in h.path:
            h.send_response(401)
            h.end_headers()
            return
        h.send_response(200)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(data)))
        if disposition:
            h.send_header("Content-Disposition", disposition)
        h.end_headers()
        h.wfile.write(data)

    return handler


def redirect(location):
    def handler(h):
        h.send_response(302)
        h.send_header("Location", location)
        h.end_headers()

    return handler


@pytest.fixture
def web():
    Handler.routes, Handler.seen = {}, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    # two hostnames for the same server, to test cross-host redirects
    yield Handler.routes, Handler.seen, f"http://127.0.0.1:{port}", f"http://localhost:{port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def small_checkpoints(monkeypatch):
    monkeypatch.setattr(cu, "MIN_CHECKPOINT_BYTES", 100)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "token", "expected"),
    [
        ("https://civitai.com/api/download/models/12", "", "https://civitai.com/api/download/models/12"),
        ("https://civitai.com/api/download/models/12?type=Model&token=old", "abc",
         "https://civitai.com/api/download/models/12?type=Model&token=abc"),
        ("https://civitai.com/models/7/cool?modelVersionId=99", "abc",
         "https://civitai.com/api/download/models/99?type=Model&format=SafeTensor&token=abc"),
        ("https://civitai.com/api/download/models/12?token=mine", "",  # pasted token is kept
         "https://civitai.com/api/download/models/12?token=mine"),
        ("https://civitai.com/models/7?modelVersionId=99&token=mine", "",
         "https://civitai.com/api/download/models/99?type=Model&format=SafeTensor&token=mine"),
    ],
)  # fmt: skip
def test_civitai_download_url(url, token, expected):
    assert cu.civitai_download_url(url, token) == expected


def test_civitai_url_without_version_is_rejected():
    with pytest.raises(cu.DownloadError, match="modelVersionId"):
        cu.civitai_download_url("https://civitai.com/models/7/cool")


def test_download_names_file_and_uses_part_file(web, tmp_path):
    routes, _, a, _ = web
    routes["/f"] = send(CKPT, disposition='attachment; filename="my model.safetensors"')
    path = cu.download(a + "/f", tmp_path)
    assert path == tmp_path / "my model.safetensors" and path.read_bytes() == CKPT
    assert not list(tmp_path.glob("*.part"))
    assert cu.download(a + "/f", tmp_path) == path  # complete file is reused


def test_filename_from_signed_url_and_path(web, tmp_path):
    routes, _, a, _ = web
    disposition = quote('attachment; filename="../../evil.safetensors"')
    routes["/s"] = redirect(f"{a}/cdn?response-content-disposition={disposition}")
    routes["/cdn"] = send(CKPT)
    assert cu.download(a + "/s", tmp_path).name == "evil.safetensors"  # path parts stripped
    routes["/plain.safetensors"] = send(CKPT)
    assert cu.download(a + "/plain.safetensors", tmp_path).name == "plain.safetensors"


def test_civitai_token_reaches_civitai_but_not_storage(web, tmp_path):
    routes, seen, a, b = web
    routes["/api/download/models/1"] = redirect(f"{b}/storage?signed=1")
    routes["/storage"] = send(CKPT, disposition='attachment; filename="c.safetensors"')
    url = cu.civitai_download_url(a + "/api/download/models/1", "secret")
    assert cu.download(url, tmp_path).name == "c.safetensors"
    assert [path for path, _ in seen] == [
        "/api/download/models/1?token=secret",
        "/storage?signed=1",
    ]


def test_errors_never_echo_the_token(web, tmp_path):
    _, _, a, _ = web
    dead = f"http://127.0.0.1:{free_port()}"
    for url in (a + "/missing?token=secret", dead + "/x?token=secret"):
        with pytest.raises(cu.DownloadError) as info:
            cu.download(url, tmp_path)
        assert "secret" not in str(info.value) and info.value.__cause__ is None


@pytest.mark.parametrize(
    ("route", "match"),
    [(send(b"<html>", content_type="text/html"), "web page"), (send(CKPT, auth=True), "token")],
)
def test_download_errors(web, tmp_path, route, match):
    routes, _, a, _ = web
    routes["/f"] = route
    with pytest.raises(cu.DownloadError, match=match):
        cu.download(a + "/f", tmp_path)


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("m.ckpt", CKPT, ".safetensors"),
        ("m.safetensors", CKPT[:50], "too small"),
        ("m.safetensors", b"<!DOCTYPE html>" + b"x" * 200, "not a valid"),
        ("m.safetensors", struct.pack("<Q", 20) + b"not json" * 20, "not a valid"),
    ],
)  # fmt: skip
def test_validate_checkpoint_rejects(tmp_path, name, content, match):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(cu.DownloadError, match=match):
        cu.validate_checkpoint(path)


def test_fetch_checkpoint_routes_by_host(web, tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    calls = {}

    def fake_hf(**kwargs):
        calls.update(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(CKPT)
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", fake_hf)
    path = cu.fetch_checkpoint(
        "https://huggingface.co/org/repo/blob/main/m.safetensors", tmp_path, hf_token="hf_x"
    )
    assert path == tmp_path / "m.safetensors"
    assert (calls["repo_id"], calls["revision"], calls["token"]) == ("org/repo", "main", "hf_x")

    routes, _, a, _ = web
    routes["/direct.safetensors"] = send(CKPT)
    assert cu.fetch_checkpoint(a + "/direct.safetensors", tmp_path).name == "direct.safetensors"
    for bad in ("", "not a url", "https://huggingface.co/org/repo"):
        with pytest.raises(cu.DownloadError):
            cu.fetch_checkpoint(bad, tmp_path)


# ---------------------------------------------------------------------------
# Secrets, install, server
# ---------------------------------------------------------------------------


def test_get_token_precedence(monkeypatch):
    monkeypatch.delenv("SDMT_TEST_TOKEN", raising=False)
    assert cu.get_token("SDMT_TEST_TOKEN") == ""
    monkeypatch.setenv("SDMT_TEST_TOKEN", "env")
    assert cu.get_token("SDMT_TEST_TOKEN") == "env"
    assert cu.get_token("SDMT_TEST_TOKEN", " typed ") == "typed"  # notebook value wins
    assert cu.get_token("SDMT_TEST_TOKEN", "?token=abc123") == "abc123"  # pasted URL form


def test_install_requirements_warns_if_torch_changes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cu, "run", lambda *a, **k: None)
    versions = iter(["2.11", "2.12", "2.12"])
    monkeypatch.setattr(cu, "version", lambda name: next(versions))
    cu.install_requirements(tmp_path)
    assert "pip changed torch 2.11 → 2.12" in capsys.readouterr().out


def test_run_reports_output_on_failure():
    with pytest.raises(RuntimeError, match="boom"):
        cu.run([sys.executable, "-c", "print('boom'); raise SystemExit(1)"], "test")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_start_restart_stop(tmp_path):
    port = free_port()
    work = tmp_path / "work"  # server.log / server.pid land here, not in the repo
    work.mkdir()
    cmd = ["env", f"PYTHONPATH={REPO_ROOT}", sys.executable, "-m", "src.cli", "serve", "--mock",
           "--model-family", "sd15", "--port", str(port)]  # fmt: skip
    try:
        cu.start_server(cmd, cwd=work, port=port)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            assert r.status == 200
        first_pid = (work / "server.pid").read_text()
        cu.start_server(cmd, cwd=work, port=port)  # restart replaces the old process
        assert (work / "server.pid").read_text() != first_pid and cu.port_open(port)
        assert "Server online" in (work / "server.log").read_text()  # logs go to the file
    finally:
        cu.stop_server(work / "server.pid")
    assert not cu.port_open(port) and not (work / "server.pid").exists()
    cu.stop_server(work / "server.pid")  # idempotent


def test_server_start_failure_shows_log(tmp_path):
    with pytest.raises(RuntimeError, match="bad config"):
        cu.start_server([sys.executable, "-c", "print('bad config')"], cwd=tmp_path,
                        port=free_port())  # fmt: skip


def test_show_ui_outside_colab(capsys):
    cu.show_ui(8000)
    assert "127.0.0.1:8000" in capsys.readouterr().out


def test_urls_without_scheme_are_accepted(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cu, "download", lambda url, d: seen.append(url) or tmp_path / "x")
    monkeypatch.setattr(cu, "validate_checkpoint", lambda path: None)
    cu.fetch_checkpoint("civitai.com/models/1?modelVersionId=2", tmp_path)
    assert seen == ["https://civitai.com/api/download/models/2?type=Model&format=SafeTensor"]


def test_huggingface_errors_are_short_with_token_hint(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")

    class GatedRepoError(Exception):
        pass

    def gated(**kwargs):
        raise GatedRepoError("401 Client Error ... long library message")

    monkeypatch.setattr(hub, "hf_hub_download", gated)
    with pytest.raises(cu.DownloadError, match="set HF_TOKEN") as info:
        cu.fetch_checkpoint("https://huggingface.co/o/r/blob/main/m.safetensors", tmp_path)
    assert info.value.__cause__ is None and "long library message" not in str(info.value)

