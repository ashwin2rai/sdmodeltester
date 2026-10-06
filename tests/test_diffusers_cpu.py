"""Real DiffusersBackend on CPU with tiny random SD1.5/SDXL pipelines.

No checkpoint, no network: models are built from small configs and the CLIP tokenizer
from an in-memory byte-level vocab. Requires the optional inference stack::

    uv sync --group inference-cpu

Skipped automatically when torch/diffusers are not installed. What these tests cannot
cover (single-file loading, CUDA, fp16, speed, image quality) is left to Phase 8 on Colab.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

import src.backend  # noqa: E402
from src.backend import (  # noqa: E402
    SAMPLER_IDS,
    DiffusersBackend,
    GenerationError,
    make_scheduler,
    resolve_job,
)
from src.prompting import encode_prompt_batch  # noqa: E402
from tests.test_seeds import make_request  # noqa: E402

pytestmark = pytest.mark.torch

SIZE = 64  # tiny models: keep CPU time low (validation minimum lowered below)


@pytest.fixture(autouse=True)
def allow_tiny_sizes(monkeypatch):
    monkeypatch.setattr(src.backend, "MIN_DIMENSION", 64)


# ---------------------------------------------------------------------------
# Tiny pipeline builders
# ---------------------------------------------------------------------------


def tiny_tokenizer(pad_token="<|endoftext|>"):
    from transformers import CLIPTokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)}
    vocab.update({c + "</w>": len(chars) + i for i, c in enumerate(chars)})
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    return CLIPTokenizer(vocab=vocab, merges=[], model_max_length=77, pad_token=pad_token)


def tiny_text_config(**extra):
    return transformers.CLIPTextConfig(
        bos_token_id=512,
        eos_token_id=513,
        pad_token_id=513,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=5,
        vocab_size=1000,
        **extra,
    )


def tiny_vae():
    return diffusers.AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
    )


def tiny_scheduler():
    return diffusers.EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        timestep_spacing="leading",
    )


def build_sd15():
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
    )
    pipe = diffusers.StableDiffusionPipeline(
        vae=tiny_vae(),
        text_encoder=transformers.CLIPTextModel(tiny_text_config()),
        tokenizer=tiny_tokenizer(),
        unet=unet,
        scheduler=tiny_scheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return pipe.to("cpu")


def build_sdxl():
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
        cross_attention_dim=64,
        norm_num_groups=1,
    )
    pipe = diffusers.StableDiffusionXLPipeline(
        vae=tiny_vae(),
        text_encoder=transformers.CLIPTextModel(tiny_text_config()),
        text_encoder_2=transformers.CLIPTextModelWithProjection(
            tiny_text_config(projection_dim=32)
        ),
        tokenizer=tiny_tokenizer(),
        tokenizer_2=tiny_tokenizer(pad_token="!"),
        unet=unet,
        scheduler=tiny_scheduler(),
    )
    return pipe.to("cpu")


BUILDERS = {"sd15": build_sd15, "sdxl": build_sdxl}


def make_backend(family, **kwargs):
    logs = []
    backend = DiffusersBackend(
        family,
        f"tiny-{family}.safetensors",
        device="cpu",
        dtype="float32",
        warmup_steps=1,
        warmup_size=(SIZE, SIZE),
        log=logs.append,
        pipeline_loader=BUILDERS[family],
        **kwargs,
    )
    backend.logs = logs
    return backend


@pytest.fixture(scope="module", params=["sd15", "sdxl"])
def loaded(request):
    backend = make_backend(request.param)
    states = []
    backend.load(lambda state, msg: states.append(state))
    backend.warmup(lambda state, msg: states.append(state))
    backend.states = states
    return backend


def tiny_request(**overrides):
    fields = dict(
        prompt="a cat",
        negative_prompt="blurry",
        width=SIZE,
        height=SIZE,
        steps=2,
        sampler="euler",
        seed=1,
    )
    fields.update(overrides)
    return make_request(**fields)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_and_warmup(loaded, tmp_path):
    assert loaded.states == ["loading", "warming"]  # baseline: no optimizing step
    assert loaded.txt2img is not None and loaded.img2img is not None
    assert any("Warm-up complete" in line for line in loaded.logs)


def test_img2img_shares_weights(loaded):
    for name in ("unet", "vae", "text_encoder"):
        assert getattr(loaded.img2img, name) is getattr(loaded.txt2img, name)
    if loaded.family == "sdxl":
        assert loaded.img2img.text_encoder_2 is loaded.txt2img.text_encoder_2


def test_safety_checker_and_watermark_disabled(loaded):
    for pipe in (loaded.txt2img, loaded.img2img):
        if loaded.family == "sd15":
            assert pipe.safety_checker is None
        else:
            assert getattr(pipe, "watermark", None) is None


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

EXPECTED = {
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", "dpmsolver++", True),
    "dpmpp_2m_sde_karras": ("DPMSolverMultistepScheduler", "sde-dpmsolver++", True),
    "euler": ("EulerDiscreteScheduler", None, False),
    "euler_a": ("EulerAncestralDiscreteScheduler", None, None),
    "heun": ("HeunDiscreteScheduler", None, False),
    "dpm2_karras": ("KDPM2DiscreteScheduler", None, True),
}


@pytest.mark.parametrize("sampler", SAMPLER_IDS)
def test_scheduler_mapping(sampler):
    # Start from a karras-enabled config to prove overrides win over checkpoint config.
    base = diffusers.DPMSolverMultistepScheduler(use_karras_sigmas=True).config
    scheduler = make_scheduler(sampler, base)
    cls, algorithm, karras = EXPECTED[sampler]
    assert type(scheduler).__name__ == cls
    if algorithm:
        assert scheduler.config.algorithm_type == algorithm
    if karras is not None:
        assert scheduler.config.use_karras_sigmas is karras


@pytest.mark.parametrize("sampler", SAMPLER_IDS)
def test_every_sampler_generates(loaded, tmp_path, sampler):
    job = resolve_job(tiny_request(sampler=sampler, seed=3))
    result = loaded.generate(job, tmp_path)
    assert len(result.output_paths) == 1


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------


def test_unweighted_matches_diffusers_encode_prompt(loaded):
    pipe = loaded.txt2img
    ours = encode_prompt_batch(pipe, loaded.family, ["a red cat"], ["blurry"])
    with torch.no_grad():
        if loaded.family == "sd15":
            ref, ref_neg = pipe.encode_prompt("a red cat", "cpu", 1, True, "blurry")
        else:
            ref, ref_neg, ref_pooled, ref_neg_pooled = pipe.encode_prompt(
                "a red cat", device="cpu", negative_prompt="blurry"
            )
            assert torch.allclose(ours["pooled_prompt_embeds"], ref_pooled, atol=1e-5)
            assert torch.allclose(ours["negative_pooled_prompt_embeds"], ref_neg_pooled, atol=1e-5)
    assert ours["prompt_embeds"].shape == ref.shape
    assert torch.allclose(ours["prompt_embeds"], ref, atol=1e-5)
    assert torch.allclose(ours["negative_prompt_embeds"], ref_neg, atol=1e-5)


def test_weight_changes_only_weighted_prompt(loaded):
    pipe = loaded.txt2img
    plain = encode_prompt_batch(pipe, loaded.family, ["a red cat"], [""])["prompt_embeds"]
    unit = encode_prompt_batch(pipe, loaded.family, ["a (red:1.0) cat"], [""])["prompt_embeds"]
    heavy = encode_prompt_batch(pipe, loaded.family, ["a (red:1.5) cat"], [""])["prompt_embeds"]
    assert torch.allclose(plain, unit, atol=1e-6)
    assert not torch.allclose(plain, heavy, atol=1e-3)
    # Mean is restored after weighting (A1111 "original" emphasis).
    assert torch.allclose(plain.mean(), heavy.mean(), atol=1e-5)


def test_long_and_mixed_prompts_share_sequence_length(loaded):
    long_prompt = "word " * 40  # 160 single-char tokens with the tiny tokenizer
    out = encode_prompt_batch(loaded.txt2img, loaded.family, ["short", long_prompt], ["", ""])
    assert out["prompt_embeds"].shape[:2] == (2, 77 * 3)
    assert out["negative_prompt_embeds"].shape == out["prompt_embeds"].shape


def test_identical_prompts_encoded_once(loaded, monkeypatch):
    calls = []
    encoder = loaded.txt2img.text_encoder
    original = encoder.forward

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "forward", counting)
    encode_prompt_batch(loaded.txt2img, loaded.family, ["a cat"] * 5, ["blurry"] * 5)
    assert len(calls) == 2  # one positive + one negative, not ten


def test_sdxl_empty_negative_is_zeros():
    backend = make_backend("sdxl")
    backend.load()
    out = encode_prompt_batch(backend.txt2img, "sdxl", ["a cat", "a dog"], ["", "blurry"])
    assert torch.count_nonzero(out["negative_prompt_embeds"][0]) == 0
    assert torch.count_nonzero(out["negative_pooled_prompt_embeds"][0]) == 0
    assert torch.count_nonzero(out["negative_prompt_embeds"][1]) > 0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_txt2img_batch_writes_ordered_pngs_with_progress(loaded, tmp_path):
    events = []
    job = resolve_job(tiny_request(seed=10, num_images=3, steps=3), job_id=5)
    result = loaded.generate(job, tmp_path, lambda f, m: events.append((f, m)))
    assert result.seeds == (10, 11, 12)
    assert [p.name.split("_")[-1] for p in result.output_paths] == [
        "seed10.png",
        "seed11.png",
        "seed12.png",
    ]
    for path in result.output_paths:
        with Image.open(path) as img:
            assert img.size == (SIZE, SIZE) and img.format == "PNG"
            assert img.text == {}
    assert len(events) == 3 and events[-1] == (pytest.approx(1.0), "denoising 3/3")


def test_image_depends_only_on_its_seed(loaded, tmp_path):
    """Batch composition must not change an image (so OOM splitting is invisible)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    batch = loaded.generate(resolve_job(tiny_request(seed=20, num_images=2)), tmp_path / "a")
    alone = loaded.generate(resolve_job(tiny_request(seed=21)), tmp_path / "b")
    with Image.open(batch.output_paths[1]) as x, Image.open(alone.output_paths[0]) as y:
        diff = max(abs(a - b) for a, b in zip(x.tobytes(), y.tobytes(), strict=True))
    assert diff <= 2


def test_img2img(loaded, tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (400, 300), (200, 30, 30)).save(src)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events = []
    job = resolve_job(
        tiny_request(input_image=src, strength=0.5, steps=4, num_images=2, width=64, height=80)
    )
    result = loaded.generate(job, out_dir, lambda f, m: events.append(m))
    assert len(result.output_paths) == 2
    with Image.open(result.output_paths[0]) as img:
        assert img.size == (64, 80)
    assert events[-1] == "denoising 2/2"  # int(4 * 0.5) denoising steps


def test_oom_fallback_on_real_pipeline(tmp_path):
    backend = make_backend("sd15")
    backend.load()
    original_call = type(backend.txt2img).__call__
    sizes = []

    def limited_call(self, *args, **kwargs):
        size = kwargs["prompt_embeds"].shape[0]
        sizes.append(size)
        if size > 2:
            raise torch.OutOfMemoryError("simulated")
        return original_call(self, *args, **kwargs)

    backend.txt2img.__class__ = type(
        "Limited", (type(backend.txt2img),), {"__call__": limited_call}
    )
    result = backend.generate(resolve_job(tiny_request(num_images=5, steps=1)), tmp_path)
    assert len(result.output_paths) == 5
    assert sizes == [5, 2, 2, 1]
    assert backend.batcher.limits[("txt2img", SIZE, SIZE)] == 2
    assert any("out of memory" in line for line in backend.logs)


def test_generate_before_load_fails(tmp_path):
    with pytest.raises(GenerationError):
        make_backend("sd15").generate(resolve_job(tiny_request()), tmp_path)


def test_warmup_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = make_backend("sd15")
    backend.load()
    backend.warmup()
    assert list(Path(tmp_path).rglob("*.png")) == []


# ---------------------------------------------------------------------------
# Optional acceleration fallback
# ---------------------------------------------------------------------------


def test_compile_failure_at_warmup_falls_back(monkeypatch, tmp_path):
    class Broken(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            raise RuntimeError("inductor exploded")

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    monkeypatch.setattr(torch, "compile", lambda module, **kw: Broken(module))
    backend = make_backend("sd15", optimization="compile")
    states = []
    backend.load(lambda state, msg: states.append(state))
    assert states == ["loading", "optimizing"]
    assert backend.active_optimization == "compile"
    backend.warmup()
    assert backend.active_optimization == "baseline"
    assert not isinstance(backend.txt2img.unet, Broken)
    assert backend.img2img.unet is backend.txt2img.unet
    assert any("falling back to baseline" in line for line in backend.logs)
    result = backend.generate(resolve_job(tiny_request(steps=1)), tmp_path)
    assert len(result.output_paths) == 1


def test_compile_failure_at_setup_falls_back(monkeypatch):
    def explode(module, **kwargs):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch, "compile", explode)
    backend = make_backend("sd15", optimization="compile")
    backend.load()
    assert backend.active_optimization == "baseline"
    backend.warmup()


# ---------------------------------------------------------------------------
# CLI real path (tiny pipeline injected in place of from_single_file)
# ---------------------------------------------------------------------------


def test_cli_generate_real_backend(tmp_path, monkeypatch):
    import src.cli as cli

    def tiny_backend(family, model, **kwargs):
        kwargs.update(device="cpu", dtype="float32", warmup_size=(SIZE, SIZE), warmup_steps=1)
        return DiffusersBackend(family, model, pipeline_loader=BUILDERS[family], **kwargs)

    monkeypatch.setattr(cli, "DiffusersBackend", tiny_backend)
    model = tmp_path / "tiny.safetensors"
    model.touch()
    out, err = [], []
    code = cli.main(
        [
            "generate",
            "--model-family", "sdxl",
            "--model", str(model),
            "--prompt", "a (red:1.3) {cat | dog}",
            "--width", "64", "--height", "64", "--steps", "2",
            "--images", "2", "--seed", "7",
            "--output-dir", str(tmp_path / "out"),
        ],
        out=out.append,
        err=err.append,
    )  # fmt: skip
    assert code == 0, err
    assert [Path(p).name.split("_")[-1] for p in out] == ["seed7.png", "seed8.png"]
    assert any("Loaded tiny.safetensors (sdxl)" in line for line in err)


# ---------------------------------------------------------------------------
# Phase 8 tooling against the real backend (tiny pipelines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_benchmark_and_functional_checks_real_backend(family, tmp_path):
    from src import benchmark

    small = dict(steps=2, size=(SIZE, SIZE), alt_size=(SIZE + 16, SIZE), log=lambda m: None)
    result = benchmark.benchmark_profile(make_backend(family), family, tmp_path / "perf", **small)
    assert result["profile_active"] == "baseline"
    assert result["timings_s"]["batch_10"] > 0
    assert result["peak_vram_batches"] is None  # CPU: no CUDA stats

    backend = make_backend(family)
    backend.load()
    checks = benchmark.functional_checks(backend, family, tmp_path / "func", **small)
    failed = {c.name: c.detail for c in checks if not c.ok}
    assert failed == {}
