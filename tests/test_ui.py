"""Browser tests of src/static/index.html against the mock backend (headless Chromium).

Requires the optional UI group and a browser::

    uv sync --group ui
    uv run playwright install chromium --only-shell   # + `playwright install-deps` once

Skipped automatically when Playwright or the browser is unavailable.
"""

import threading
import time

import pytest
from PIL import Image

playwright_api = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server  # noqa: E402

from src.backend import MockBackend  # noqa: E402
from src.server import ServerState, create_app  # noqa: E402
from tests.test_server import GatedBackend  # noqa: E402

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 — browser binary or system libs missing
            pytest.skip(f"Chromium unavailable: {exc}".splitlines()[0])
        yield browser
        browser.close()


class LiveServer:
    def __init__(self, backend, tmp_path):
        self.backend = backend
        self.state = ServerState(backend, tmp_path / "inputs", tmp_path / "outputs", echo=False)
        self.httpd = make_server("127.0.0.1", 0, create_app(self.state), threaded=True)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.state.start()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if isinstance(self.backend, GatedBackend):
            self.backend.load_gate.set()
            self.backend.gen_gate.set()
        self.httpd.shutdown()
        self.state.stop()


@pytest.fixture
def live(tmp_path):
    with LiveServer(MockBackend("sdxl", "model.safetensors", step_seconds=0.002), tmp_path) as s:
        yield s


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 520, "height": 1000})
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.on("console", lambda m: page.errors.append(m.text) if m.type == "error" else None)
    yield page
    context.close()


def open_ready(page, live):
    page.goto(live.url)
    page.wait_for_function("document.getElementById('status-pill-text').textContent === 'Ready'")


def text(page, selector):
    return page.text_content(selector)


def generate(page, prompt="a cat", images=1):
    page.fill("#prompt", prompt)
    page.fill("#images", str(images))
    page.dispatch_event("#images", "input")
    page.click("#generate-btn")


def wait_for_job(page, job_id):
    page.wait_for_function(
        f"document.getElementById('result-job').textContent === 'Job {job_id}'", timeout=15000
    )


def png_file(tmp_path, name="photo.png", size=(300, 200)):
    path = tmp_path / name
    Image.new("RGB", size, (30, 140, 60)).save(path)
    return path


# ---------------------------------------------------------------------------


def test_defaults_and_header_from_config(page, live):
    open_ready(page, live)
    assert text(page, "#model-info") == "SDXL · model.safetensors"
    assert page.input_value("#width") == "1024" and page.input_value("#cfg") == "5"
    assert page.input_value("#sampler") == "dpmpp_2m_sde_karras"
    assert page.locator("#sampler option").count() == 6
    assert page.input_value("#seed") == "-1"
    assert page.input_value("#image-select") == ""
    assert page.is_disabled("#strength")
    assert text(page, "#image-hint") == "Text-to-image"
    assert page.errors == []


def test_generate_label_counts_and_seed_randomize(page, live):
    open_ready(page, live)
    page.fill("#images", "10")
    page.dispatch_event("#images", "input")
    assert text(page, "#generate-btn") == "Generate 10 images"
    page.click("#randomize-seed")
    seed = int(page.input_value("#seed"))
    assert 0 <= seed <= 2**32 - 1
    page.fill("#prompt", "abc")
    page.dispatch_event("#prompt", "input")
    assert text(page, "#prompt-count") == "3"


def test_queue_during_loading_then_ready(page, tmp_path):
    backend = GatedBackend()
    with LiveServer(backend, tmp_path) as live:
        page.goto(live.url)
        page.wait_for_function(
            "document.getElementById('status-pill-text').textContent === 'Loading'"
        )
        generate(page, "while loading")
        page.wait_for_function(
            "document.getElementById('status-text').textContent.includes('waiting for model')"
        )
        assert page.is_enabled("#generate-btn")
        backend.load_gate.set()
        wait_for_job(page, 1)
        page.wait_for_function(
            "document.getElementById('status-pill-text').textContent === 'Ready'"
        )
        assert "Queued job 1 (waiting for model)" in text(page, "#log")
    assert page.errors == []


def test_ten_image_strip_and_selection(page, live):
    open_ready(page, live)
    page.fill("#seed", "100")
    generate(page, "a {red | blue} cat", images=10)
    wait_for_job(page, 1)
    assert page.locator("#strip button").count() == 10
    assert text(page, "#viewer-name") == "Image 1/10 · seed 100"
    page.click("#strip button:nth-child(4)")
    assert text(page, "#viewer-name") == "Image 4/10 · seed 103"
    assert "seed103" in page.get_attribute("#viewer-image", "src")
    assert page.get_attribute("#viewer-open", "href") == page.get_attribute("#viewer-image", "src")
    assert "selected" in page.get_attribute("#strip button:nth-child(4)", "class")
    assert text(page, "#viewer-prompt") in ("a red cat", "a blue cat")
    # natural size check: the image actually loaded from the server
    page.wait_for_function("document.getElementById('viewer-image').naturalWidth === 1024")


def test_img2img_upload_strength_and_session_persistence(page, live, tmp_path):
    open_ready(page, live)
    page.set_input_files("#upload-file", str(png_file(tmp_path)))
    page.wait_for_function("document.getElementById('image-select').value === 'photo.png'")
    assert text(page, "#image-hint") == "photo.png selected · image-to-image"
    assert page.is_enabled("#strength")
    generate(page, "restyle")
    wait_for_job(page, 1)
    assert live.state.latest_completed.job.request.strength == 0.6
    page.reload()
    page.wait_for_function("document.getElementById('image-select').value === 'photo.png'")
    page.select_option("#image-select", "")
    assert page.is_disabled("#strength")


def test_upload_rejects_non_image(page, live, tmp_path):
    open_ready(page, live)
    bad = tmp_path / "fake.png"
    bad.write_bytes(b"not an image")
    page.set_input_files("#upload-file", str(bad))
    page.wait_for_function("document.getElementById('upload-error').textContent !== ''")
    assert "not a readable image" in text(page, "#upload-error")


def test_validation_error_is_shown(page, live):
    open_ready(page, live)
    generate(page, "a {red} cat")
    page.wait_for_function("document.getElementById('queue-error').textContent !== ''")
    assert "two alternatives" in text(page, "#queue-error")
    page.fill("#prompt", "   ")
    page.click("#generate-btn")
    assert text(page, "#queue-error") == "Prompt is required"


def test_previous_outputs_and_reuse(page, live):
    open_ready(page, live)
    for job in (1, 2):
        generate(page, "x", images=4)
        wait_for_job(page, job)
    page.click("#previous-outputs summary")
    page.wait_for_function("document.querySelectorAll('#recent button').length === 5")
    assert page.locator("#output-select option").count() == 9  # placeholder + 8 files
    assert page.is_disabled("#reuse-btn")
    newest = page.get_attribute("#recent button:nth-child(1)", "title")
    page.click("#recent button:nth-child(1)")
    assert page.input_value("#output-select") == newest
    page.click("#reuse-btn")
    page.wait_for_function(f"document.getElementById('image-select').value === {newest!r}")
    assert "image-to-image" in text(page, "#image-hint")


def test_clear_queue_and_clear_all(page, tmp_path):
    backend = GatedBackend()
    with LiveServer(backend, tmp_path) as live:
        page.goto(live.url)
        for _ in range(3):
            generate(page, "x")
        page.wait_for_function(
            "document.getElementById('queue-text').textContent === '2 prompts queued'"
        )
        page.click("#danger-zone summary")
        page.once("dialog", lambda d: d.accept())
        page.click("#clear-all-btn")
        page.wait_for_function("document.getElementById('clear-all-error').textContent !== ''")
        assert "clear the queue" in text(page, "#clear-all-error")

        page.click("#clear-queue-btn")
        page.wait_for_function(
            "document.getElementById('queue-text').textContent === '0 prompts queued'"
        )
        backend.load_gate.set()
        wait_for_job(page, 1)
        page.wait_for_function("document.getElementById('status-text').textContent === 'Idle'")

        page.once("dialog", lambda d: d.dismiss())
        page.click("#clear-all-btn")
        time.sleep(0.3)
        assert len(list(live.state.outputs_dir.glob("*.png"))) == 1  # dismissed: nothing deleted

        page.once("dialog", lambda d: d.accept())
        page.click("#clear-all-btn")
        page.wait_for_function("document.getElementById('viewer-empty').hidden === false")
        assert list(live.state.outputs_dir.glob("*.png")) == []
        assert text(page, "#clear-all-error") == ""


def test_prompt_text_is_never_html(page, live):
    open_ready(page, live)
    payload = '<img src=x onerror="window.__xss=1">'
    generate(page, payload)
    wait_for_job(page, 1)
    assert payload in text(page, "#viewer-prompt")
    assert page.locator("#viewer-prompt img, #log img").count() == 0
    assert page.evaluate("window.__xss") is None


def test_offline_pill_and_reconnect(page, live):
    open_ready(page, live)
    page.route("**/api/status*", lambda route: route.abort())
    page.wait_for_function("document.getElementById('status-pill-text').textContent === 'Offline'")
    page.unroute("**/api/status*")
    page.wait_for_function("document.getElementById('status-pill-text').textContent === 'Ready'")
    assert "Reconnected" in text(page, "#log")


def test_backend_error_state(page, tmp_path):
    with LiveServer(MockBackend("sd15", fail_load=True), tmp_path) as live:
        page.goto(live.url)
        page.wait_for_function(
            "document.getElementById('status-pill-text').textContent === 'Error'"
        )
        assert "Mock checkpoint failed to load" in text(page, "#status-text")
        generate(page, "x")
        page.wait_for_function("document.getElementById('queue-error').textContent !== ''")
        assert "backend unavailable" in text(page, "#queue-error")


def test_mobile_width_has_no_horizontal_overflow(browser, live):
    context = browser.new_context(viewport={"width": 360, "height": 800})
    page = context.new_page()
    try:
        page.goto(live.url)
        page.wait_for_function(
            "document.getElementById('status-pill-text').textContent === 'Ready'"
        )
        generate(page, "x", images=10)
        wait_for_job(page, 1)
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0
        # the thumbnail strip scrolls horizontally inside the card instead
        assert page.evaluate(
            "(() => { const s = document.getElementById('strip');"
            " return s.scrollWidth > s.clientWidth; })()"
        )
    finally:
        context.close()
