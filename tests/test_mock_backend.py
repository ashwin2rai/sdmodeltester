import re
from pathlib import Path

import pytest
from PIL import Image

from src.backend import (
    GenerationError,
    MicroBatcher,
    MockBackend,
    MockOutOfMemory,
    resolve_job,
)
from tests.test_seeds import make_request

FILENAME = re.compile(r"^\d{8}_\d{6}_job(\d{4})_img(\d{2})_seed(\d+)\.png$")


@pytest.fixture
def backend():
    b = MockBackend("sd15")
    b.load()
    b.warmup()
    return b


def run(backend, tmp_path, progress=None, job_id=1, **overrides):
    job = resolve_job(make_request(**overrides), job_id=job_id)
    return job, backend.generate(job, tmp_path, progress)


def write_input(tmp_path, size=(300, 200)) -> Path:
    path = tmp_path / "input.png"
    Image.new("RGB", size, (10, 200, 30)).save(path)
    return path


# --- lifecycle ---------------------------------------------------------------


def test_load_and_warmup_report_states():
    states = []
    b = MockBackend("sdxl")
    b.load(lambda state, msg: states.append(state))
    b.warmup(lambda state, msg: states.append(state))
    assert states == ["loading", "optimizing", "warming"]


def test_fail_load_raises():
    with pytest.raises(RuntimeError):
        MockBackend("sd15", fail_load=True).load()


def test_generate_before_load_fails(tmp_path):
    job = resolve_job(make_request())
    with pytest.raises(GenerationError):
        MockBackend("sd15").generate(job, tmp_path)


def test_rejects_unknown_family():
    with pytest.raises(ValueError):
        MockBackend("auto")


# --- generation --------------------------------------------------------------


def test_single_image_txt2img(backend, tmp_path):
    job, result = run(backend, tmp_path, seed=123)
    assert result.seeds == (123,)
    assert len(result.output_paths) == 1
    path = result.output_paths[0]
    assert path.parent == tmp_path
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size == (512, 512)
    assert result.resolved_prompts == (job.images[0].prompt,)
    assert result.elapsed_seconds >= 0


def test_ten_images_in_seed_order(backend, tmp_path):
    job, result = run(backend, tmp_path, seed=500, num_images=10, job_id=7)
    assert result.seeds == tuple(range(500, 510))
    assert len(result.output_paths) == 10
    for i, path in enumerate(result.output_paths):
        m = FILENAME.match(path.name)
        assert m, path.name
        assert (int(m[1]), int(m[2]), int(m[3])) == (7, i + 1, 500 + i)
    assert backend.batch_sizes == [10]  # one true batch


def test_outputs_are_deterministic_per_seed(backend, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, a = run(backend, tmp_path / "a", seed=42)
    _, b = run(backend, tmp_path / "b", seed=42)
    with Image.open(a.output_paths[0]) as ia, Image.open(b.output_paths[0]) as ib:
        assert ia.tobytes() == ib.tobytes()


def test_img2img_uses_preprocessed_input(backend, tmp_path):
    src = write_input(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _, result = run(backend, out_dir, input_image=src, strength=0.5, width=256, height=384, seed=1)
    with Image.open(result.output_paths[0]) as img:
        assert img.size == (256, 384)
        # bottom-right corner keeps the (tinted) green input rather than a flat mock color
        r, g, b = img.getpixel((250, 380))
        assert g > r and g > b


def test_png_has_no_generation_metadata(backend, tmp_path):
    src = tmp_path / "with_meta.jpg"
    img = Image.new("RGB", (300, 300), "red")
    exif = img.getexif()
    exif[0x010E] = "secret description"
    img.save(src, exif=exif)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _, result = run(backend, out_dir, input_image=src, strength=0.6, prompt="secret prompt")
    data = result.output_paths[0].read_bytes()
    assert b"secret" not in data
    for chunk in (b"tEXt", b"iTXt", b"zTXt", b"eXIf"):
        assert chunk not in data
    with Image.open(result.output_paths[0]) as out:
        assert out.text == {}


def test_filename_collision_is_avoided(backend, tmp_path):
    _, a = run(backend, tmp_path, seed=9, job_id=3)
    _, b = run(backend, tmp_path, seed=9, job_id=3)
    if a.output_paths[0].name[:15] == b.output_paths[0].name[:15]:  # same second
        assert b.output_paths[0].name.endswith("_seed9_1.png")
    assert a.output_paths[0] != b.output_paths[0]
    assert len(list(tmp_path.glob("*.png"))) == 2


# --- progress ----------------------------------------------------------------


def test_progress_reaches_one_and_is_monotonic(backend, tmp_path):
    events = []
    run(backend, tmp_path, progress=lambda f, m: events.append((f, m)), steps=25)
    fractions = [f for f, _ in events]
    assert len(events) == 25
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert events[13][1] == "denoising 14/25"


def test_img2img_progress_counts_denoising_steps(backend, tmp_path):
    src = write_input(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events = []
    run(
        backend,
        out_dir,
        progress=lambda f, m: events.append(m),
        input_image=src,
        strength=0.6,
        steps=25,
    )
    assert len(events) == 15
    assert events[-1] == "denoising 15/15"


# --- OOM fallback / micro-batching ----------------------------------------------


def test_oom_fallback_halves_and_remembers(tmp_path):
    b = MockBackend("sdxl", max_batch=5)
    b.load()
    events = []
    _, result = run(
        b,
        tmp_path,
        progress=lambda f, m: events.append((f, m)),
        num_images=10,
        seed=100,
        width=1024,
        height=1024,
        steps=4,
    )
    assert b.batch_sizes == [10, 5, 5]
    assert result.seeds == tuple(range(100, 110))
    assert len(result.output_paths) == 10
    assert b.batcher.limits == {("txt2img", 1024, 1024): 5}
    messages = [m for _, m in events]
    assert messages[0] == "Batch 1/2 · denoising 1/4"
    assert messages[-1] == "Batch 2/2 · denoising 4/4"
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)

    # Next job at the same mode/size starts from the remembered size.
    b.batch_sizes.clear()
    run(b, tmp_path, num_images=10, width=1024, height=1024, steps=1)
    assert b.batch_sizes == [5, 5]

    # A different resolution is not limited.
    b.batch_sizes.clear()
    b.max_batch = None
    run(b, tmp_path, num_images=10, width=512, height=512, steps=1)
    assert b.batch_sizes == [10]


def test_mid_job_failure_reports_completed_files(tmp_path):
    b = MockBackend("sd15", fail_after_images=3)
    b.load()
    job = resolve_job(make_request(num_images=5, seed=1))
    with pytest.raises(GenerationError) as info:
        b.generate(job, tmp_path)
    assert len(info.value.completed_paths) == 3
    assert all(p.exists() for p in info.value.completed_paths)


def test_microbatcher_uneven_split():
    batcher = MicroBatcher()
    sizes = []

    def run_batch(batch, on_step):
        sizes.append(len(batch))
        if len(batch) > 3:
            raise MockOutOfMemory()
        on_step(1, 1)
        return list(batch)

    def is_oom(exc):
        return isinstance(exc, MockOutOfMemory)

    oom_calls = []
    out = batcher.run(
        ("txt2img", 8, 8), list(range(7)), run_batch, is_oom, on_oom=lambda *a: oom_calls.append(a)
    )
    assert out == list(range(7))
    assert sizes == [7, 3, 3, 1]
    assert oom_calls == [(7, 3)]
    assert batcher.limits[("txt2img", 8, 8)] == 3


def test_microbatcher_lowers_limit_on_later_oom():
    batcher = MicroBatcher()
    batcher.limits[("k", 1, 1)] = 4

    def run_batch(batch, on_step):
        if len(batch) > 2:
            raise MockOutOfMemory()
        return list(batch)

    out = batcher.run(("k", 1, 1), list(range(8)), run_batch, lambda e: True)
    assert out == list(range(8))
    assert batcher.limits[("k", 1, 1)] == 2


def test_microbatcher_reraises_non_oom_and_size_one_oom():
    batcher = MicroBatcher()

    def boom(batch, on_step):
        raise ValueError("not oom")

    with pytest.raises(ValueError):
        batcher.run(("k", 1, 1), [1, 2], boom, lambda e: isinstance(e, MockOutOfMemory))

    def always_oom(batch, on_step):
        raise MockOutOfMemory()

    with pytest.raises(MockOutOfMemory):
        batcher.run(("k", 1, 1), [1, 2, 3, 4], always_oom, lambda e: True)
