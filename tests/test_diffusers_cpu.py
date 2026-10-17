"""Real DiffusersBackend on CPU with tiny random SD1.5/SDXL/Z-Image pipelines.

No checkpoint, no network: models are built from small configs and the CLIP/Qwen tokenizers
from an in-memory byte-level vocab. Z-Image goes through the real single-file loader: a tiny
transformer is written as a ComfyUI-layout BF16/FP8 file next to a local components folder.
Requires the optional inference stack::

    uv sync --group inference-cpu

Skipped automatically when torch/diffusers are not installed. What these tests cannot
cover (single-file loading, CUDA, fp16, speed, image quality) is left to Phase 8 on Colab.
"""

import functools
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

import src.backend  # noqa: E402
from src.backend import (  # noqa: E402
    FAMILIES,
    SAMPLER_IDS,
    DiffusersBackend,
    GenerationError,
    make_scheduler,
    resolve_job,
)
from src.prompting import encode_prompt_batch  # noqa: E402
from tests.test_request import write_safetensors  # noqa: E402
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


def make_backend(family, model=None, **kwargs):
    logs = []
    options = dict(
        device="cpu",
        dtype="float32",
        warmup_steps=1,
        warmup_size=(SIZE, SIZE),
        pipeline_loader=BUILDERS.get(family),
    )
    backend = DiffusersBackend(
        family, model or f"tiny-{family}.safetensors", log=logs.append, **{**options, **kwargs}
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


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_load_keeps_half_precision(family):
    """Regression: ``from_pipe`` defaults to float32 and would upcast the shared modules."""
    half = lambda: BUILDERS[family]().to(torch.float16)  # noqa: E731
    backend = make_backend(family, dtype="float16", pipeline_loader=half)
    backend.load()
    for pipe in (backend.txt2img, backend.img2img):
        assert {pipe.unet.dtype, pipe.vae.dtype, pipe.text_encoder.dtype} == {torch.float16}


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
    "flow_euler": ("FlowMatchEulerDiscreteScheduler", None, None),
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


@pytest.mark.parametrize("sampler", FAMILIES["sd15"].samplers)  # sd15 and sdxl share the list
def test_every_sampler_generates(loaded, tmp_path, sampler):
    job = resolve_job(tiny_request(sampler=sampler, seed=3), family="sd15")
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
    job = resolve_job(tiny_request(seed=10, num_images=3, steps=3), job_id=5, family="sd15")
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
    batch = loaded.generate(
        resolve_job(tiny_request(seed=20, num_images=2), family="sd15"), tmp_path / "a"
    )
    alone = loaded.generate(resolve_job(tiny_request(seed=21), family="sd15"), tmp_path / "b")
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
        tiny_request(input_image=src, strength=0.5, steps=4, num_images=2, width=64, height=80),
        family="sd15",
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
    result = backend.generate(
        resolve_job(tiny_request(num_images=5, steps=1), family="sd15"), tmp_path
    )
    assert len(result.output_paths) == 5
    assert sizes == [5, 2, 2, 1]
    assert backend.batcher.limits[("txt2img", SIZE, SIZE)] == 2
    assert any("out of memory" in line for line in backend.logs)


def test_generate_before_load_fails(tmp_path):
    with pytest.raises(GenerationError):
        make_backend("sd15").generate(resolve_job(tiny_request(), family="sd15"), tmp_path)


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
    result = backend.generate(resolve_job(tiny_request(steps=1), family="sd15"), tmp_path)
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


# ---------------------------------------------------------------------------
# Z-Image: real single-file loader (BF16, plain FP8, ComfyUI scaled FP8) on a tiny model
# ---------------------------------------------------------------------------


def tiny_qwen_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    for special in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        vocab[special] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return tokenizer


def build_zimage():
    torch.manual_seed(0)
    tokenizer = tiny_qwen_tokenizer()
    text_encoder = transformers.Qwen3Model(
        transformers.Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=37,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=1024,
        )
    )  # fmt: skip
    vae = diffusers.AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 2,
        up_block_types=["UpDecoderBlock2D"] * 2,
        latent_channels=16,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    transformer = diffusers.ZImageTransformer2DModel(
        in_channels=16, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        cap_feat_dim=32, axes_dims=[4, 6, 6], axes_lens=[1024, 32, 32],
    )  # fmt: skip
    return diffusers.ZImagePipeline(
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )


def comfy_state_dict(transformer):
    """Diffusers -> ComfyUI key layout (inverse of diffusers' Z-Image single-file converter)."""
    sd = transformer.state_dict()
    renames = {
        "all_final_layer.2-1.": "final_layer.",
        "all_x_embedder.2-1.": "x_embedder.",
        ".attention.to_out.0.": ".attention.out.",
        ".attention.norm_k.": ".attention.k_norm.",
        ".attention.norm_q.": ".attention.q_norm.",
    }
    out = {}
    for key, value in sd.items():
        if ".attention.to_k." in key or ".attention.to_v." in key:
            continue
        if ".attention.to_q.weight" in key:
            stem = key.removesuffix(".to_q.weight")
            qkv = [sd[f"{stem}.to_{x}.weight"] for x in "qkv"]
            out[f"{stem}.qkv.weight"] = torch.cat(qkv)
            continue
        for old, new in renames.items():
            key = key.replace(old, new)
        out[key] = value.contiguous()
    return out


ZIMAGE_FILES = {"bf16": torch.bfloat16, "fp8_e4m3fn": torch.float8_e4m3fn,
                "fp8_e5m2": torch.float8_e5m2}  # fmt: skip


@pytest.fixture(scope="module")
def zimage_files(tmp_path_factory):
    from safetensors.torch import save_file

    root = tmp_path_factory.mktemp("zimage")
    pipe = build_zimage()
    pipe.save_pretrained(root / "components")
    weights = comfy_state_dict(pipe.transformer)
    for name, dtype in ZIMAGE_FILES.items():
        save_file({k: v.to(dtype) for k, v in weights.items()}, root / f"z_{name}.safetensors")
    save_file(comfy_scaled_fp8(weights), root / "z_scaled_fp8.safetensors")
    return root


def comfy_scaled_fp8(weights):
    """ComfyUI quantized layout: linear weights as FP8 / scale (+ weight_scale, comfy_quant)."""
    config = list(b'{"format": "float8_e4m3fn"}')
    out = {}
    for key, value in weights.items():
        if value.ndim != 2 or not key.endswith(".weight"):
            out[key] = value.to(torch.bfloat16)
            continue
        scale = value.abs().max().float() / 448  # e4m3 max
        out[key] = (value / scale).to(torch.float8_e4m3fn)
        out[f"{key}_scale"] = scale
        out[key.removesuffix("weight") + "comfy_quant"] = torch.tensor(config, dtype=torch.uint8)
    return out


def make_zimage_backend(zimage_files, name="bf16", **kwargs):
    components = str(zimage_files / "components")
    path = zimage_files / f"z_{name}.safetensors"
    return make_backend("zimage", path, dtype=None, zimage_components=components, **kwargs)


@pytest.fixture(scope="module")
def load_zimage(zimage_files):
    """Loaded + warmed backend per weight file, built once and shared across tests."""

    @functools.cache
    def load(name):
        backend = make_zimage_backend(zimage_files, name)
        backend.load()
        backend.warmup()
        return backend

    return load


@pytest.fixture(params=list(ZIMAGE_FILES))
def zimage(request, load_zimage):
    return load_zimage(request.param)


def zimage_request(**overrides):
    return tiny_request(**{"sampler": "flow_euler", "guidance_scale": 0.0, **overrides})


def test_zimage_loads_with_storage_dtype_and_shared_img2img(zimage):
    storage = ZIMAGE_FILES[zimage.model_path.stem.removeprefix("z_")]
    # FP8 files stay FP8 in memory; norms/embedders (precision-sensitive) are BF16.
    assert {p.dtype for p in zimage.txt2img.transformer.parameters()} == {storage, torch.bfloat16}
    assert zimage.txt2img.text_encoder.dtype == torch.bfloat16
    assert type(zimage.img2img).__name__ == "ZImageImg2ImgPipeline"
    for name in ("transformer", "vae", "text_encoder"):
        assert getattr(zimage.img2img, name) is getattr(zimage.txt2img, name)
    label = str(storage).removeprefix("torch.")
    assert any(f"Transformer weights: {label}" in line for line in zimage.logs)


def test_zimage_scaled_fp8_is_dequantized(load_zimage, tmp_path):
    backend = load_zimage("scaled_fp8")
    loaded = backend.txt2img.transformer.state_dict()
    path = Path(backend.zimage_components) / "transformer"
    original = diffusers.ZImageTransformer2DModel.from_pretrained(path).state_dict()
    assert {v.dtype for v in loaded.values()} == {torch.bfloat16}
    for key in ("layers.0.attention.to_q.weight", "layers.0.feed_forward.w1.weight"):
        error = (loaded[key].float() - original[key]).norm() / original[key].norm()
        assert error < 0.1  # FP8 rounding only; a missing scale would be off by ~100x
    assert any("scaled FP8, dequantized to bfloat16" in line for line in backend.logs)
    result = backend.generate(resolve_job(zimage_request(steps=2), family="zimage"), tmp_path)
    assert len(result.output_paths) == 1


def test_zimage_txt2img_batch(zimage, tmp_path):
    events = []
    job = resolve_job(zimage_request(seed=10, num_images=2, steps=3), family="zimage")
    result = zimage.generate(job, tmp_path, lambda f, m: events.append(m))
    assert result.seeds == (10, 11)
    for path in result.output_paths:
        with Image.open(path) as img:
            assert img.size == (SIZE, SIZE)
    assert events[-1] == "denoising 3/3"


def test_zimage_img2img(zimage, tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (400, 300), (200, 30, 30)).save(src)
    (tmp_path / "out").mkdir()
    events = []
    req = zimage_request(input_image=src, strength=0.5, steps=4, width=64, height=80)
    result = zimage.generate(
        resolve_job(req, family="zimage"), tmp_path / "out", lambda f, m: events.append(m)
    )
    with Image.open(result.output_paths[0]) as img:
        assert img.size == (64, 80)
    assert events[-1] == "denoising 2/2"  # int(4 * 0.5) denoising steps


def test_zimage_cfg_uses_negative_prompt(zimage, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    def run(neg, out):
        req = zimage_request(guidance_scale=4.0, negative_prompt=neg)
        return zimage.generate(resolve_job(req, family="zimage"), tmp_path / out).output_paths[0]

    with Image.open(run("", "a")) as x, Image.open(run("blurry, dark", "b")) as y:
        assert x.tobytes() != y.tobytes()


def test_zimage_load_rejects_unsupported_checkpoint(tmp_path):
    sd = write_safetensors(tmp_path / "sd.safetensors", {"input_blocks.0.0.weight": "F16"})
    with pytest.raises(ValueError, match="not a Z-Image transformer"):
        make_backend("zimage", sd, dtype=None).load()


def test_zimage_compile_failure_falls_back(zimage_files, monkeypatch, tmp_path):
    def explode(module, **kwargs):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch, "compile", explode)
    backend = make_zimage_backend(zimage_files, optimization="compile")
    backend.load()
    assert backend.active_optimization == "baseline"
    assert backend.img2img.transformer is backend.txt2img.transformer
    result = backend.generate(resolve_job(zimage_request(steps=1), family="zimage"), tmp_path)
    assert len(result.output_paths) == 1


def test_zimage_functional_checks(load_zimage, tmp_path):
    from src import benchmark

    backend = load_zimage("fp8_e4m3fn")
    small = dict(steps=2, size=(SIZE, SIZE), alt_size=(SIZE + 16, SIZE), log=lambda m: None)
    checks = benchmark.functional_checks(backend, "zimage", tmp_path, **small)
    assert {c.name: c.detail for c in checks if not c.ok} == {}
    names = [c.name for c in checks]
    assert "sampler Flow Euler" in names and "weighted prompt" not in names
