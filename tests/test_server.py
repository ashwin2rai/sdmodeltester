import io
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

import src.cli as cli
from src import server
from src.backend import MockBackend
from src.server import ServerState, create_app, secure_delete

REPO_ROOT = Path(__file__).resolve().parents[1]


class GatedBackend(MockBackend):
    """Mock backend whose load and generate block until the test releases them."""

    def __init__(self, family="sdxl", **kwargs):
        super().__init__(family, "gated.safetensors", **kwargs)
        self.load_gate = threading.Event()
        self.gen_gate = threading.Event()
        self.gen_gate.set()
        self.generating = threading.Event()

    def load(self, status_callback=None):
        assert self.load_gate.wait(10), "test never released load"
        super().load(status_callback)

    def generate(self, job, output_dir, progress_callback=None):
        self.generating.set()
        assert self.gen_gate.wait(10), "test never released generate"
        return super().generate(job, output_dir, progress_callback)


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


@pytest.fixture
def env(tmp_path):
    backend = GatedBackend()
    state = ServerState(backend, tmp_path / "inputs", tmp_path / "outputs", echo=False)
    client = create_app(state).test_client()
    state.start()
    yield state, client, backend
    backend.load_gate.set()
    backend.gen_gate.set()
    state.stop()


def ready_env(env):
    state, client, backend = env
    backend.load_gate.set()
    wait_until(lambda: state.backend_state == "ready")
    return state, client, backend


def queue(client, **fields):
    body = {"prompt": "a cat", "width": 256, "height": 256, "steps": 2, **fields}
    return client.post("/api/queue", json=body)


def png_bytes(size=(32, 24), color="red"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def upload(client, data, name="photo.png"):
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# Startup / readiness
# ---------------------------------------------------------------------------


def test_ui_and_api_respond_before_backend_ready(env):
    state, client, _ = env
    assert client.get("/").status_code == 200
    config = client.get("/api/config").get_json()
    assert config["model_family"] == "sdxl" and config["model_name"] == "gated.safetensors"
    assert config["defaults"]["width"] == 1024 and config["defaults"]["guidance_scale"] == 5.0
    assert [s["id"] for s in config["samplers"]][:2] == ["dpmpp_2m_karras", "dpmpp_2m_sde_karras"]
    status = client.get("/api/status").get_json()
    assert status["backend_state"] in ("starting", "loading")
    assert status["current_job"] is None and status["latest_completed_job"] is None


def test_enqueue_while_loading_then_runs_when_ready(env):
    state, client, backend = env
    response = queue(client, seed=123, num_images=3)
    assert response.status_code == 200
    body = response.get_json()
    assert body["job_id"] == 1 and body["seeds"] == [123, 124, 125]
    assert body["waiting_for_model"] is True

    wait_until(lambda: state.current is not None)
    current = client.get("/api/status").get_json()["current_job"]
    assert current == {
        "id": 1,
        "status": "waiting_for_model",
        "message": "Waiting for model…",
        "progress": 0.0,
        "num_images": 3,
    }

    backend.load_gate.set()
    wait_until(lambda: state.latest_completed is not None)
    status = client.get("/api/status").get_json()
    assert status["backend_state"] == "ready"
    latest = status["latest_completed_job"]
    assert latest["id"] == 1 and latest["seeds"] == [123, 124, 125]
    assert [name.split("_")[-1] for name in latest["outputs"]] == [
        "seed123.png",
        "seed124.png",
        "seed125.png",
    ]
    for name in latest["outputs"]:
        response = client.get(f"/outputs/{name}")
        assert response.status_code == 200 and response.mimetype == "image/png"


def test_status_reports_progress_while_running(env):
    state, client, backend = ready_env(env)
    backend.gen_gate.clear()
    queue(client)
    wait_until(backend.generating.is_set)
    current = client.get("/api/status").get_json()["current_job"]
    assert current["status"] == "running"
    backend.gen_gate.set()
    wait_until(lambda: state.latest_completed is not None)


def test_queue_capacity_enforced(env):
    state, client, _ = env
    assert queue(client).status_code == 200  # becomes the waiting active job
    wait_until(lambda: state.current is not None)
    for _ in range(server.MAX_QUEUE):
        assert queue(client).status_code == 200
    response = queue(client)
    assert response.status_code == 429
    assert "full" in response.get_json()["error"]
    assert client.get("/api/status").get_json()["queue_length"] == server.MAX_QUEUE


def test_clear_queue_keeps_active_job(env):
    state, client, backend = env
    for _ in range(3):
        queue(client)
    wait_until(lambda: state.current is not None and state.queue.qsize() == 2)
    assert client.delete("/api/queue").get_json() == {"cleared": 2}
    assert state.current.id == 1
    backend.load_gate.set()
    wait_until(lambda: state.latest_completed is not None)
    time.sleep(0.2)
    assert state.latest_completed.id == 1
    assert state.current is None
    assert len(list(state.outputs_dir.glob("*.png"))) == 1  # jobs 2 and 3 never ran


def test_enqueue_response_includes_corrections_and_prompts(env):
    _, client, _ = ready_env(env)
    body = queue(client, width=516, seed=5, num_images=2, prompt="a {red | blue} cat").get_json()
    assert body["corrections"] == ["width 516 rounded to 520 (multiple of 8)"]
    assert all(p in ("a red cat", "a blue cat") for p in body["prompts"])
    assert body["waiting_for_model"] is False


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"prompt": ""}, "prompt is required"),
        ({"prompt": "a {red} cat"}, "two alternatives"),
        ({"prompt": "x", "num_images": 11}, "image count"),
        ({"prompt": "x", "width": "512"}, "integer"),
        ({"prompt": "x", "sampler": "ddim"}, "sampler"),
        ({"prompt": "x", "input_image": "missing.png"}, "input image not found"),
        ({"prompt": "x", "input_image": "../etc/passwd"}, "input image not found"),
    ],
)
def test_enqueue_validation_errors(env, body, fragment):
    _, client, _ = env
    response = client.post("/api/queue", json=body)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_enqueue_rejects_non_object_body(env):
    _, client, _ = env
    assert client.post("/api/queue", json=[1, 2]).status_code == 400
    assert client.post("/api/queue", data="nope").status_code == 400


def test_img2img_job_uses_uploaded_input(env):
    state, client, _ = ready_env(env)
    name = upload(client, png_bytes((300, 200), "green")).get_json()["filename"]
    response = queue(client, input_image=name, strength=None)  # null -> family default
    assert response.status_code == 200
    wait_until(lambda: state.latest_completed is not None)
    assert state.latest_completed.job.request.strength == 0.6
    assert state.latest_completed.job.request.input_image == state.inputs_dir / name


def test_strength_ignored_for_txt2img(env):
    state, client, _ = ready_env(env)
    assert queue(client, strength=0.4).status_code == 200
    wait_until(lambda: state.latest_completed is not None)
    assert state.latest_completed.job.request.strength is None


# ---------------------------------------------------------------------------
# Load failure
# ---------------------------------------------------------------------------


def test_fatal_load_error(tmp_path):
    backend = GatedBackend(fail_load=True)
    state = ServerState(backend, tmp_path / "i", tmp_path / "o", echo=False)
    client = create_app(state).test_client()
    state.start()
    try:
        queue(client)
        queue(client)
        wait_until(lambda: state.current is not None)
        backend.load_gate.set()
        wait_until(lambda: state.backend_state == "error")
        wait_until(lambda: state.current is None)
        status = client.get("/api/status").get_json()
        assert status["backend_state"] == "error"
        assert "Mock checkpoint failed to load" in status["backend_error"]
        assert status["queue_length"] == 0
        texts = [entry["text"] for entry in status["log"]]
        assert any("Backend error" in t for t in texts)
        assert any("Failed 1 pending job" in t for t in texts)
        assert any("Job 1: failed" in t for t in texts)
        response = queue(client)
        assert response.status_code == 503
        assert client.get("/").status_code == 200  # UI stays up for diagnosis
    finally:
        state.stop()


def test_generation_error_is_not_latest_completed(tmp_path):
    backend = GatedBackend(fail_after_images=1)
    backend.load_gate.set()
    state = ServerState(backend, tmp_path / "i", tmp_path / "o", echo=False)
    client = create_app(state).test_client()
    state.start()
    try:
        queue(client, num_images=3)
        wait_until(lambda: any("error" in t["text"] for t in state.log_since(0)))
        wait_until(lambda: state.current is None)
        assert state.latest_completed is None
        texts = [t["text"] for t in state.log_since(0)]
        assert any("1 image(s) completed before the error" in t for t in texts)
        assert len(list(state.outputs_dir.glob("*.png"))) == 1  # partial work kept
    finally:
        state.stop()


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


def test_log_is_incremental(env):
    state, client, _ = ready_env(env)
    first = client.get("/api/status").get_json()["log"]
    assert first and first[-1]["text"].startswith("Backend: ready")
    last_seq = first[-1]["seq"]
    queue(client, seed=7)
    new = client.get(f"/api/status?log_after={last_seq}").get_json()["log"]
    assert new[0]["text"] == "Queued job 1: 1 image(s), seed 7"
    assert all(entry["seq"] > last_seq for entry in new)


def test_format_seeds():
    assert server.format_seeds((5,)) == "seed 5"
    assert server.format_seeds((123, 124, 125)) == "seeds 123–125"
    assert server.format_seeds((9, 3, 7)) == "seeds 9, 3, 7"
    assert server.format_seeds((1, 9, 3, 7, 5)) == "seeds 1, 9, 3, 7, …"


# ---------------------------------------------------------------------------
# Files: upload, listing, serving, reuse
# ---------------------------------------------------------------------------


def test_upload_validation(env):
    state, client, _ = env
    assert client.post("/api/upload").status_code == 400
    assert upload(client, b"not an image", "fake.png").status_code == 400
    assert upload(client, png_bytes(), "notes.txt").status_code == 400
    assert list(state.inputs_dir.iterdir()) == []  # rejected uploads leave nothing behind

    first = upload(client, png_bytes(), "my photo.png").get_json()["filename"]
    second = upload(client, png_bytes(), "my photo.png").get_json()["filename"]
    assert first == "my_photo.png" and second == "my_photo_1.png"
    traversal = upload(client, png_bytes(), "../../evil.png").get_json()["filename"]
    assert traversal == "evil.png" and (state.inputs_dir / "evil.png").is_file()


def test_listings_newest_first_and_filtered(env):
    state, client, _ = env
    for i, name in enumerate(["old.png", "mid.jpg", "new.webp"]):
        path = state.inputs_dir / name
        path.write_bytes(png_bytes())
        os.utime(path, (1000 + i, 1000 + i))
    (state.inputs_dir / "notes.txt").write_text("x")
    (state.inputs_dir / ".gitkeep").touch()
    assert client.get("/api/inputs").get_json() == {"files": ["new.webp", "mid.jpg", "old.png"]}

    for i, name in enumerate(["a.png", "b.png"]):
        path = state.outputs_dir / name
        path.write_bytes(png_bytes())
        os.utime(path, (2000 - i, 2000 - i))
    (state.outputs_dir / "c.jpg").write_bytes(png_bytes())
    assert client.get("/api/outputs").get_json() == {"files": ["a.png", "b.png"]}


def test_file_serving_is_confined(env, tmp_path):
    state, client, _ = env
    (state.inputs_dir / "in.png").write_bytes(png_bytes())
    (tmp_path / "secret.png").write_bytes(png_bytes())
    assert client.get("/inputs/in.png").status_code == 200
    assert client.get("/inputs/missing.png").status_code == 404
    assert client.get("/inputs/..%2Fsecret.png").status_code == 404
    assert client.get("/outputs/..%2Fsecret.png").status_code == 404
    assert client.get("/outputs/in.png").status_code == 404


def test_reuse_output_copies_into_inputs(env):
    state, client, _ = ready_env(env)
    queue(client)
    wait_until(lambda: state.latest_completed is not None)
    name = state.latest_completed.outputs[0]
    response = client.post("/api/reuse-output", json={"filename": name})
    assert response.status_code == 200
    copied = response.get_json()["filename"]
    assert copied == name
    assert (state.inputs_dir / copied).read_bytes() == (state.outputs_dir / name).read_bytes()
    assert (state.outputs_dir / name).is_file()  # original kept
    again = client.post("/api/reuse-output", json={"filename": name}).get_json()["filename"]
    assert again != copied and again.endswith("_1.png")
    assert client.get("/api/inputs").get_json()["files"][0] == again

    for bad in ("missing.png", "../inputs/x.png", None, 5):
        assert client.post("/api/reuse-output", json={"filename": bad}).status_code == 404


def test_mtimes_change_with_directory_contents(env):
    state, client, _ = env
    before = client.get("/api/status").get_json()["inputs_mtime"]
    time.sleep(0.02)
    upload(client, png_bytes())
    after = client.get("/api/status").get_json()["inputs_mtime"]
    assert after > before


# ---------------------------------------------------------------------------
# Clear all
# ---------------------------------------------------------------------------


def test_clear_all_rejected_while_active_or_queued(env):
    state, client, backend = env
    queue(client)
    wait_until(lambda: state.current is not None)
    response = client.delete("/api/clear-all")
    assert response.status_code == 409 and "queue" in response.get_json()["error"]
    queue(client)
    assert client.delete("/api/clear-all").status_code == 409
    backend.load_gate.set()
    wait_until(lambda: state.current is None and state.queue.empty())
    assert client.delete("/api/clear-all").status_code == 200


def test_clear_all_pending_only_rejected(env):
    state, client, backend = ready_env(env)
    backend.gen_gate.clear()
    queue(client)
    wait_until(backend.generating.is_set)
    queue(client)
    client.delete("/api/queue")
    assert client.delete("/api/clear-all").status_code == 409  # active job still running
    backend.gen_gate.set()
    wait_until(lambda: state.current is None)
    assert client.delete("/api/clear-all").status_code == 200


def test_clear_all_when_idle(env):
    state, client, _ = ready_env(env)
    upload(client, png_bytes())
    queue(client, num_images=2)
    wait_until(lambda: state.latest_completed is not None and state.current is None)
    (state.inputs_dir / ".gitkeep").touch()
    (state.outputs_dir / "sub").mkdir()
    response = client.delete("/api/clear-all")
    assert response.get_json() == {"deleted": 3}
    assert sorted(p.name for p in state.inputs_dir.iterdir()) == [".gitkeep"]
    assert sorted(p.name for p in state.outputs_dir.iterdir()) == ["sub"]
    assert client.get("/api/status").get_json()["latest_completed_job"] is None


def test_secure_delete_overwrites_before_unlink(tmp_path, monkeypatch):
    path = tmp_path / "f.png"
    original = b"SECRET" * 1000
    path.write_bytes(original)
    seen = {}
    real_unlink = Path.unlink

    def spy(self, *args, **kwargs):
        seen["bytes"] = self.read_bytes()
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", spy)
    secure_delete(path)
    assert not path.exists()
    assert len(seen["bytes"]) == len(original) and seen["bytes"] != original


# ---------------------------------------------------------------------------
# Wiring: CLI serve, real HTTP, no torch
# ---------------------------------------------------------------------------


def test_cli_serve_passes_arguments(monkeypatch, tmp_path):
    captured = {}

    def fake_serve(backend, **kwargs):
        captured.update(kwargs, backend=backend)

    monkeypatch.setattr(server, "serve", fake_serve)
    code = cli.main(
        [
            "serve",
            "--mock",
            "--model-family",
            "sd15",
            "--port",
            "8123",
            "--inputs-dir",
            str(tmp_path / "in"),
            "--mock-load-seconds",
            "0.5",
        ],
        out=lambda m: None,
        err=lambda m: None,
    )
    assert code == 0
    assert captured["port"] == 8123 and captured["host"] == "127.0.0.1"
    assert captured["inputs_dir"] == tmp_path / "in"
    assert captured["backend"].family == "sd15" and captured["backend"].load_seconds == 0.5


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_over_real_http(tmp_path):
    port = free_port()
    backend = MockBackend("sd15", step_seconds=0.0)
    thread = threading.Thread(
        target=server.serve,
        kwargs=dict(
            backend=backend,
            port=port,
            inputs_dir=tmp_path / "i",
            outputs_dir=tmp_path / "o",
        ),
        daemon=True,
    )
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=2) as response:
            return response.status, response.read()

    wait_until(lambda: _reachable(base), timeout=10)
    assert get("/")[0] == 200
    wait_until(lambda: b'"ready"' in get("/api/status")[1], timeout=10)


def _reachable(base):
    try:
        urllib.request.urlopen(base + "/api/config", timeout=0.5).close()
        return True
    except OSError:
        return False


def test_server_import_does_not_pull_torch():
    code = "import sys, src.server; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
