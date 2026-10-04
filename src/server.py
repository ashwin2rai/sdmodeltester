"""Flask server: queue, backend-loader thread, GPU worker thread, filesystem API.

The HTTP server starts immediately; the model loads/optimizes/warms in a background
thread (SPEC §13.5, §16). Jobs may be queued at any time before a fatal load error and
run one at a time on a single worker.
"""

from __future__ import annotations

import itertools
import logging
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

from src.backend import (
    FAMILY_DEFAULTS,
    SAMPLER_LABELS,
    SAMPLERS,
    Backend,
    GenerationError,
    GenerationRequest,
    ResolvedGenerationJob,
    ValidationError,
    collision_safe_path,
    resolve_job,
)

MAX_QUEUE = 5
LOG_LINES = 300
MAX_UPLOAD_BYTES = 50 * 2**20
INPUT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OUTPUT_EXTENSIONS = {".png"}
PRESERVED_FILES = {".gitkeep"}  # clear-all leaves the repo placeholder in place
STATIC_DIR = Path(__file__).resolve().parent / "static"

ACCEPTING_STATES = {"starting", "loading", "optimizing", "warming", "ready"}


# ---------------------------------------------------------------------------
# Job records and server state
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    id: int
    job: ResolvedGenerationJob
    status: str = "queued"  # queued | waiting_for_model | running | done | error | cancelled
    message: str = "Queued"
    progress: float = 0.0
    outputs: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "num_images": self.job.request.num_images,
        }


def format_seeds(seeds: tuple[int, ...]) -> str:
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    if seeds == tuple(range(seeds[0], seeds[0] + len(seeds))):
        return f"seeds {seeds[0]}–{seeds[-1]}"
    shown = ", ".join(map(str, seeds[:4]))
    return f"seeds {shown}{', …' if len(seeds) > 4 else ''}"


class ServerState:
    """All mutable server state; guarded by ``lock``."""

    def __init__(
        self,
        backend: Backend,
        inputs_dir: Path,
        outputs_dir: Path,
        *,
        max_queue: int = MAX_QUEUE,
        echo: bool = True,
    ) -> None:
        self.backend = backend
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo

        self.lock = threading.RLock()
        self.queue: queue.Queue[JobRecord] = queue.Queue(maxsize=max_queue)
        self.ready = threading.Event()  # set when backend is ready *or* failed
        self.stopping = threading.Event()
        self.backend_state = "starting"
        self.backend_message = "Starting…"
        self.backend_error: str | None = None
        self.current: JobRecord | None = None
        self.latest_completed: JobRecord | None = None
        self._job_ids = itertools.count(1)
        self._log: deque[tuple[int, str]] = deque(maxlen=LOG_LINES)
        self._log_seq = 0
        self._threads: list[threading.Thread] = []

        if hasattr(backend, "log"):
            backend.log = self.log  # route real-backend diagnostics into the shared log

    # -- logging ---------------------------------------------------------------

    def log(self, text: str) -> None:
        with self.lock:
            self._log_seq += 1
            self._log.append((self._log_seq, text))
        if self.echo:
            print(f"[{datetime.now():%H:%M:%S}] {text}", flush=True)

    def log_since(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [{"seq": seq, "text": text} for seq, text in self._log if seq > after]

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        for target, name in ((self._worker, "gpu-worker"), (self._loader, "backend-loader")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        self.stopping.set()
        self.ready.set()
        for thread in self._threads:
            thread.join(timeout)

    def _set_backend_status(self, state: str, message: str) -> None:
        with self.lock:
            self.backend_state = state
            self.backend_message = message
        self.log(f"Backend: {message}")

    def _loader(self) -> None:
        started = time.monotonic()
        try:
            self.backend.load(self._set_backend_status)
            self.backend.warmup(self._set_backend_status)
        except Exception as exc:  # noqa: BLE001 — any load failure is fatal for the session
            traceback.print_exc()
            self._fail_backend(f"{type(exc).__name__}: {exc}")
            return
        with self.lock:
            self.backend_state = "ready"
            self.backend_message = "Ready"
        self.log(f"Backend: ready ({time.monotonic() - started:.1f}s)")
        self.ready.set()

    def _fail_backend(self, error: str) -> None:
        with self.lock:
            self.backend_state = "error"
            self.backend_message = "Backend failed to load"
            self.backend_error = error
            drained = self._drain_queue("failed")
        self.log(f"Backend error: {error}")
        if drained:
            self.log(f"Failed {drained} pending job(s): backend unavailable")
        self.ready.set()

    # -- queue -------------------------------------------------------------------

    def enqueue(self, req: GenerationRequest) -> JobRecord:
        """Resolve seeds/prompts now and add to the pending queue (raises on refusal)."""
        with self.lock:
            if self.backend_state not in ACCEPTING_STATES:
                raise BackendUnavailable(self.backend_error or "backend unavailable")
            if self.queue.full():
                raise QueueFull(f"queue is full ({self.queue.maxsize} pending jobs)")
            job_id = next(self._job_ids)
            record = JobRecord(job_id, resolve_job(req, job_id=job_id))
            self.queue.put_nowait(record)
            waiting = "" if self.backend_state == "ready" else " (waiting for model)"
        r = record.job.request
        self.log(
            f"Queued job {job_id}{waiting}: {r.num_images} image(s), "
            f"{format_seeds(record.job.seeds)}"
        )
        for note in record.job.corrections:
            self.log(f"Job {job_id}: {note}")
        return record

    def _drain_queue(self, status: str) -> int:
        count = 0
        while True:
            try:
                record = self.queue.get_nowait()
            except queue.Empty:
                return count
            record.status = status
            record.message = "Cleared" if status == "cancelled" else "Backend unavailable"
            count += 1

    def clear_queue(self) -> int:
        with self.lock:
            count = self._drain_queue("cancelled")
        self.log(f"Cleared {count} pending job(s)")
        return count

    def is_busy(self) -> bool:
        with self.lock:
            return self.current is not None or not self.queue.empty()

    # -- worker ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self.stopping.is_set():
            try:
                record = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                if record.status != "queued":
                    continue
                self.current = record
                if not self.ready.is_set():
                    record.status = "waiting_for_model"
                    record.message = "Waiting for model…"
            self.ready.wait()
            try:
                if self.stopping.is_set():
                    return
                self._run(record)
            finally:
                with self.lock:
                    self.current = None

    def _run(self, record: JobRecord) -> None:
        if self.backend_state == "error":
            record.status, record.message = "error", "Backend unavailable"
            record.error = self.backend_error
            self.log(f"Job {record.id}: failed — backend unavailable")
            return

        req = record.job.request
        mode = req.mode
        extra = f" · strength {req.strength}" if req.input_image else ""
        with self.lock:
            record.status, record.message = "running", "Starting…"
        self.log(
            f"Job {record.id}: {mode} · {SAMPLER_LABELS[req.sampler]} · {req.steps} steps · "
            f"{req.width}x{req.height} · CFG {req.guidance_scale}{extra}"
        )
        if any(spec.prompt != req.prompt for spec in record.job.images):
            for spec in record.job.images:
                self.log(f"Job {record.id}: image {spec.index + 1} prompt: {spec.prompt}")

        def progress(fraction: float, message: str) -> None:
            with self.lock:
                record.progress = fraction
                record.message = message

        try:
            result = self.backend.generate(record.job, self.outputs_dir, progress)
        except GenerationError as exc:
            self._job_failed(record, str(exc), exc.completed_paths)
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._job_failed(record, f"{type(exc).__name__}: {exc}", ())
            return

        with self.lock:
            record.outputs = [path.name for path in result.output_paths]
            record.status, record.message, record.progress = "done", "Done", 1.0
            self.latest_completed = record
        for name in record.outputs:
            self.log(f"Job {record.id}: saved outputs/{name}")
        self.log(f"Job {record.id}: done in {result.elapsed_seconds:.1f}s")

    def _job_failed(self, record: JobRecord, error: str, completed: tuple[Path, ...]) -> None:
        with self.lock:
            record.status, record.message, record.error = "error", "Failed", error
            record.outputs = [path.name for path in completed]
        self.log(f"Job {record.id}: error — {error}")
        if completed:
            self.log(
                f"Job {record.id}: {len(completed)} image(s) completed before the error: "
                + ", ".join(path.name for path in completed)
            )

    # -- status ------------------------------------------------------------------

    def status(self, log_after: int = 0) -> dict[str, Any]:
        with self.lock:
            latest = self.latest_completed
            return {
                "backend_state": self.backend_state,
                "backend_message": self.backend_message,
                "backend_error": self.backend_error,
                "queue_length": self.queue.qsize(),
                "current_job": self.current.summary() if self.current else None,
                "latest_completed_job": (
                    {
                        "id": latest.id,
                        "outputs": latest.outputs,
                        "seeds": list(latest.job.seeds),
                        "prompts": [spec.prompt for spec in latest.job.images],
                    }
                    if latest
                    else None
                ),
                "outputs_mtime": _dir_mtime(self.outputs_dir),
                "inputs_mtime": _dir_mtime(self.inputs_dir),
                "log": self.log_since(log_after),
            }


class BackendUnavailable(Exception):
    pass


class QueueFull(Exception):
    pass


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _dir_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_files(directory: Path, extensions: set[str]) -> list[str]:
    """Regular files with an allowed extension, newest first."""
    entries = []
    for path in directory.iterdir():
        if path.suffix.lower() in extensions and path.is_file() and not path.is_symlink():
            entries.append((path.stat().st_mtime_ns, path.name))
    return [name for _, name in sorted(entries, reverse=True)]


def safe_child(directory: Path, name: Any, extensions: set[str]) -> Path | None:
    """Resolve a client-supplied bare filename inside ``directory``, or None."""
    if not isinstance(name, str) or not name or secure_filename(name) != name:
        return None
    path = directory / name
    if path.suffix.lower() not in extensions or not path.is_file() or path.is_symlink():
        return None
    return path


def secure_delete(path: Path) -> None:
    """Best-effort overwrite-before-unlink. Cannot defeat SSD wear levelling, snapshots,
    journaling, or cloud infrastructure copies."""
    size = path.stat().st_size
    with open(path, "r+b") as fh:
        remaining = size
        chunk = 1 << 20
        while remaining > 0:
            n = min(chunk, remaining)
            fh.write(os.urandom(n))
            remaining -= n
        fh.flush()
        os.fsync(fh.fileno())
    path.unlink()


def clear_directory(directory: Path) -> int:
    count = 0
    for path in directory.iterdir():
        if path.name in PRESERVED_FILES or path.is_symlink() or not path.is_file():
            continue
        secure_delete(path)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

REQUEST_FIELDS = {
    "width",
    "height",
    "steps",
    "guidance_scale",
    "sampler",
    "seed",
    "num_images",
    "strength",
}


def request_from_json(data: Any, family: str, inputs_dir: Path) -> GenerationRequest:
    if not isinstance(data, dict):
        raise ValidationError("request body must be a JSON object")
    d = FAMILY_DEFAULTS[family]
    prompt = data.get("prompt", "")
    negative = data.get("negative_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("prompt is required")

    input_image = None
    strength = None
    name = data.get("input_image")
    if name not in (None, ""):
        input_image = safe_child(inputs_dir, name, INPUT_EXTENSIONS)
        if input_image is None:
            raise ValidationError(f"input image not found: {name}")
        strength = data.get("strength")
        if strength is None:
            strength = d.strength

    def get(key: str, default: Any) -> Any:
        value = data.get(key)
        return default if value is None else value

    return GenerationRequest(
        prompt=prompt,
        negative_prompt=negative,
        width=get("width", d.width),
        height=get("height", d.height),
        steps=get("steps", d.steps),
        guidance_scale=get("guidance_scale", d.guidance_scale),
        sampler=get("sampler", d.sampler),
        seed=get("seed", d.seed),
        num_images=get("num_images", d.num_images),
        input_image=input_image,
        strength=strength,
    )


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def create_app(state: ServerState) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    family = state.backend.family

    @app.get("/")
    def index():
        response = send_from_directory(STATIC_DIR, "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/config")
    def config():
        d = FAMILY_DEFAULTS[family]
        return jsonify(
            {
                "model_family": family,
                "model_name": state.backend.model_name,
                "defaults": {
                    "width": d.width,
                    "height": d.height,
                    "steps": d.steps,
                    "guidance_scale": d.guidance_scale,
                    "sampler": d.sampler,
                    "seed": d.seed,
                    "num_images": d.num_images,
                    "strength": d.strength,
                },
                "samplers": [{"id": sid, "label": label} for sid, label in SAMPLERS],
                "max_queue": state.queue.maxsize,
            }
        )

    @app.get("/api/status")
    def status():
        try:
            log_after = int(request.args.get("log_after", 0))
        except ValueError:
            log_after = 0
        return jsonify(state.status(log_after))

    @app.get("/api/inputs")
    def inputs():
        return jsonify({"files": list_files(state.inputs_dir, INPUT_EXTENSIONS)})

    @app.get("/api/outputs")
    def outputs():
        return jsonify({"files": list_files(state.outputs_dir, OUTPUT_EXTENSIONS)})

    @app.get("/inputs/<name>")
    def input_file(name: str):
        if safe_child(state.inputs_dir, name, INPUT_EXTENSIONS) is None:
            return _error("not found", 404)
        return send_from_directory(state.inputs_dir.resolve(), name)

    @app.get("/outputs/<name>")
    def output_file(name: str):
        if safe_child(state.outputs_dir, name, OUTPUT_EXTENSIONS) is None:
            return _error("not found", 404)
        return send_from_directory(state.outputs_dir.resolve(), name)

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return _error("no file uploaded (field 'file')", 400)
        name = secure_filename(file.filename) or "upload.png"
        if Path(name).suffix.lower() not in INPUT_EXTENSIONS:
            return _error(f"unsupported file type: {Path(name).suffix or 'none'}", 400)
        # Stream to a hidden temp name (not listed: wrong suffix), verify, then publish.
        tmp = state.inputs_dir / f".upload-{uuid.uuid4().hex}.part"
        file.save(tmp)
        try:
            with Image.open(tmp) as img:
                img.verify()
        except Exception:  # noqa: BLE001 — anything Pillow can't read is rejected
            tmp.unlink(missing_ok=True)
            return _error("file is not a readable image", 400)
        with state.lock:
            dest = collision_safe_path(state.inputs_dir, name)
            tmp.rename(dest)
        state.log(f"Uploaded {dest.name}")
        return jsonify({"filename": dest.name})

    @app.post("/api/reuse-output")
    def reuse_output():
        data = request.get_json(silent=True) or {}
        source = safe_child(state.outputs_dir, data.get("filename"), OUTPUT_EXTENSIONS)
        if source is None:
            return _error("output not found", 404)
        with state.lock:
            dest = collision_safe_path(state.inputs_dir, source.name)
            shutil.copyfile(source, dest)
        state.log(f"Copied outputs/{source.name} to inputs/{dest.name}")
        return jsonify({"filename": dest.name})

    @app.post("/api/queue")
    def enqueue():
        try:
            req = request_from_json(request.get_json(silent=True), family, state.inputs_dir)
            record = state.enqueue(req)
        except ValidationError as exc:
            return _error(str(exc), 400)
        except BackendUnavailable as exc:
            return _error(f"backend unavailable: {exc}", 503)
        except QueueFull as exc:
            return _error(str(exc), 429)
        job = record.job
        return jsonify(
            {
                "job_id": record.id,
                "seeds": list(job.seeds),
                "prompts": [spec.prompt for spec in job.images],
                "corrections": list(job.corrections),
                "queue_length": state.queue.qsize(),
                "waiting_for_model": not state.ready.is_set(),
            }
        )

    @app.delete("/api/queue")
    def clear_queue():
        return jsonify({"cleared": state.clear_queue()})

    @app.delete("/api/clear-all")
    def clear_all():
        with state.lock:
            if state.current is not None or not state.queue.empty():
                return _error(
                    "a job is running or queued; clear the queue and wait for the active "
                    "job to finish first",
                    409,
                )
            deleted = clear_directory(state.inputs_dir) + clear_directory(state.outputs_dir)
            state.latest_completed = None
        state.log(f"Cleared all inputs and outputs ({deleted} file(s), best effort)")
        return jsonify({"deleted": deleted})

    return app


def serve(
    backend: Backend,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    inputs_dir: Path = Path("inputs"),
    outputs_dir: Path = Path("outputs"),
) -> None:
    """Start worker + loader threads and run the HTTP server (blocks)."""
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # no per-poll request lines
    state = ServerState(backend, inputs_dir, outputs_dir)
    app = create_app(state)
    state.log(
        f"Server online at http://{host}:{port}; "
        f"{backend.family} · {backend.model_name} loading in background."
    )
    state.start()
    app.run(host=host, port=port, threaded=True, use_reloader=False)
