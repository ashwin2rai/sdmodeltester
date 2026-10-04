"""Prompt processing: one-level dynamic alternatives and (later) weighted embeddings.

Dynamic syntax::

    a man with {white | black | silver} hair
    a man with {(white:1.2) | black} hair

Rules (SPEC §9.1): one brace level only, at least two non-empty alternatives
per group, alternatives are whitespace-trimmed, several groups per prompt are
allowed. Choices are made with a PRNG derived from the image seed (SPEC §9.2),
never with the diffusion generator.

Weighted syntax (SPEC §9.3)::

    (white hair:1.2)    (background:0.7)

Only explicit numeric ``(text:weight)`` is part of the contract. Anything else in
parentheses is literal text; ``\\(`` and ``\\)`` are literal parentheses. Weighting is
implemented here (A1111 "original" emphasis: scale token embeddings by weight, then
restore the chunk mean) rather than through a third-party helper, so the Colab
dependency surface stays at diffusers/transformers. Prompts longer than 75 tokens are
split into 77-token chunks whose embeddings are concatenated. Only the
``encode_prompt_batch`` function touches torch, and it imports it lazily.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from typing import Any

# Fixed salt so the prompt PRNG stream is independent of the diffusion noise
# stream even though both are derived from the same concrete seed.
PROMPT_RNG_SALT = 0x5D_9E_17_C3

Segment = str | tuple[str, ...]


class PromptSyntaxError(ValueError):
    """Raised when a prompt template has invalid dynamic-brace syntax."""


def parse_template(text: str) -> list[Segment]:
    """Split a template into literal strings and tuples of alternatives."""
    segments: list[Segment] = []
    literal: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "}":
            raise PromptSyntaxError(f"Unmatched '}}' at position {i}")
        if ch != "{":
            literal.append(ch)
            i += 1
            continue

        end = text.find("}", i + 1)
        if end == -1:
            raise PromptSyntaxError(f"Unclosed '{{' at position {i}")
        body = text[i + 1 : end]
        if "{" in body:
            raise PromptSyntaxError(f"Nested braces are not supported (position {i})")
        alternatives = tuple(part.strip() for part in body.split("|"))
        if len(alternatives) < 2:
            raise PromptSyntaxError(
                f"Group {{{body}}} needs at least two alternatives separated by '|'"
            )
        if any(not alt for alt in alternatives):
            raise PromptSyntaxError(f"Group {{{body}}} contains an empty alternative")

        if literal:
            segments.append("".join(literal))
            literal = []
        segments.append(alternatives)
        i = end + 1

    if literal:
        segments.append("".join(literal))
    return segments


def validate_template(text: str) -> None:
    """Raise PromptSyntaxError if the template is invalid."""
    parse_template(text)


def has_dynamic_groups(text: str) -> bool:
    return any(isinstance(seg, tuple) for seg in parse_template(text))


def prompt_rng(seed: int) -> random.Random:
    """Deterministic PRNG for dynamic choices of the image with this concrete seed."""
    return random.Random(seed ^ PROMPT_RNG_SALT)


def resolve_template(text: str, rng: random.Random) -> str:
    """Pick one alternative per group, consuming ``rng`` left to right."""
    return "".join(
        rng.choice(seg) if isinstance(seg, tuple) else seg for seg in parse_template(text)
    )


def resolve_prompt_pair(prompt: str, negative_prompt: str, seed: int) -> tuple[str, str]:
    """Resolve positive then negative template for one image."""
    rng = prompt_rng(seed)
    return resolve_template(prompt, rng), resolve_template(negative_prompt, rng)


# ---------------------------------------------------------------------------
# Weighted prompts: parsing and token chunking (pure Python)
# ---------------------------------------------------------------------------

CHUNK_TOKENS = 75  # content tokens per CLIP window (77 minus BOS/EOS)

_WEIGHT_RE = re.compile(r"\(([^()]*):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)")
_ESCAPED_OPEN, _ESCAPED_CLOSE = "\x00", "\x01"


def parse_weighted(text: str) -> list[tuple[str, float]]:
    """Split a resolved prompt into ``(text, weight)`` runs.

    ``"a (white hair:1.2) cat"`` -> ``[("a ", 1.0), ("white hair", 1.2), (" cat", 1.0)]``.
    Adjacent runs with equal weight are merged; empty runs are dropped.
    """
    masked = text.replace("\\(", _ESCAPED_OPEN).replace("\\)", _ESCAPED_CLOSE)
    runs: list[tuple[str, float]] = []

    def add(chunk: str, weight: float) -> None:
        chunk = chunk.replace(_ESCAPED_OPEN, "(").replace(_ESCAPED_CLOSE, ")")
        if not chunk:
            return
        if runs and runs[-1][1] == weight:
            runs[-1] = (runs[-1][0] + chunk, weight)
        else:
            runs.append((chunk, weight))

    pos = 0
    for match in _WEIGHT_RE.finditer(masked):
        add(masked[pos : match.start()], 1.0)
        add(match.group(1), float(match.group(2)))
        pos = match.end()
    add(masked[pos:], 1.0)
    return runs


def weighted_token_ids(
    text: str, tokenize: Callable[[str], list[int]]
) -> tuple[list[int], list[float]]:
    """Tokenize each weighted run (without special tokens) and give each token its weight."""
    ids: list[int] = []
    weights: list[float] = []
    for chunk, weight in parse_weighted(text):
        chunk_ids = tokenize(chunk)
        ids.extend(chunk_ids)
        weights.extend([weight] * len(chunk_ids))
    return ids, weights


def chunk_count(num_tokens: int) -> int:
    return max(1, math.ceil(num_tokens / CHUNK_TOKENS))


def chunk_tokens(
    ids: Sequence[int],
    weights: Sequence[float],
    *,
    bos: int,
    eos: int,
    pad: int,
    num_chunks: int | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Split into 77-token windows ``[BOS] + ≤75 tokens + [EOS] + [PAD]...``.

    ``num_chunks`` pads with empty windows so prompts in one batch share a sequence length.
    Special and padding tokens get weight 1.0.
    """
    needed = chunk_count(len(ids))
    num_chunks = needed if num_chunks is None else num_chunks
    if num_chunks < needed:
        raise ValueError(f"prompt needs {needed} chunks, only {num_chunks} allowed")

    chunks = []
    for c in range(num_chunks):
        part_ids = list(ids[c * CHUNK_TOKENS : (c + 1) * CHUNK_TOKENS])
        part_w = list(weights[c * CHUNK_TOKENS : (c + 1) * CHUNK_TOKENS])
        padding = CHUNK_TOKENS - len(part_ids)
        chunks.append(
            (
                [bos, *part_ids, eos] + [pad] * padding,
                [1.0, *part_w, 1.0] + [1.0] * padding,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Weighted prompts: embedding (torch, imported lazily)
# ---------------------------------------------------------------------------


def _tokenize_fn(tokenizer: Any) -> Callable[[str], list[int]]:
    return lambda text: tokenizer(text, add_special_tokens=False).input_ids


def _apply_weights(z: Any, weights: Any) -> Any:
    """A1111 'original' emphasis, per chunk: scale by weight, then restore the mean."""
    if bool((weights == 1.0).all()):
        return z
    original_mean = z.mean(dim=(1, 2), keepdim=True)
    z = z * weights.unsqueeze(-1).to(z.dtype)
    new_mean = z.mean(dim=(1, 2), keepdim=True)
    return z * (original_mean / new_mean.where(new_mean.abs() > 1e-12, original_mean))


def encode_prompt_batch(
    pipe: Any, family: str, prompts: Sequence[str], negative_prompts: Sequence[str]
) -> dict[str, Any]:
    """Return ``prompt_embeds`` / ``negative_prompt_embeds`` (+ pooled for SDXL) kwargs.

    Each distinct string is encoded once; all prompts in the batch are padded to the
    same number of 77-token chunks. Unweighted prompts of ≤75 tokens produce exactly the
    embeddings of the pipeline's own ``encode_prompt`` (SD1.5: last hidden state; SDXL:
    penultimate hidden states of both encoders, concatenated, plus pooled output of the
    second encoder).
    """
    import torch

    sdxl = family == "sdxl"
    encoders = [(pipe.tokenizer, pipe.text_encoder)]
    if sdxl:
        encoders.append((pipe.tokenizer_2, pipe.text_encoder_2))

    texts = list(dict.fromkeys([*prompts, *negative_prompts]))
    tokens = {
        (i, text): weighted_token_ids(text, _tokenize_fn(tok))
        for i, (tok, _) in enumerate(encoders)
        for text in texts
    }
    num_chunks = max(chunk_count(len(ids)) for ids, _ in tokens.values())
    device = pipe.text_encoder.device
    dtype = pipe.unet.dtype

    embeds: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    with torch.no_grad():
        for text in texts:
            parts = []
            for i, (tok, encoder) in enumerate(encoders):
                ids, weights = tokens[(i, text)]
                chunks = chunk_tokens(
                    ids,
                    weights,
                    bos=tok.bos_token_id,
                    eos=tok.eos_token_id,
                    pad=tok.pad_token_id,
                    num_chunks=num_chunks,
                )
                input_ids = torch.tensor([c[0] for c in chunks], device=device)
                weight_t = torch.tensor([c[1] for c in chunks], device=device)
                out = encoder(input_ids, output_hidden_states=True)
                z = out.hidden_states[-2] if sdxl else out.last_hidden_state
                z = _apply_weights(z, weight_t)
                parts.append(z.reshape(1, -1, z.shape[-1]))  # concat chunks along sequence
                if sdxl and i == 1:
                    pooled[text] = out[0][:1]  # projected pooled output of the first chunk
            embeds[text] = torch.cat(parts, dim=-1).to(dtype)

    def stack(items: Sequence[str]) -> Any:
        return torch.cat([embeds[t] for t in items])

    kwargs = {"prompt_embeds": stack(prompts), "negative_prompt_embeds": stack(negative_prompts)}
    if sdxl:
        neg_pooled = [pooled[t] for t in negative_prompts]
        if getattr(pipe.config, "force_zeros_for_empty_prompt", False):
            # Match diffusers: an empty SDXL negative prompt means zero embeddings.
            for row, text in enumerate(negative_prompts):
                if text == "":
                    kwargs["negative_prompt_embeds"][row].zero_()
                    neg_pooled[row] = torch.zeros_like(neg_pooled[row])
        kwargs["pooled_prompt_embeds"] = torch.cat([pooled[t] for t in prompts]).to(dtype)
        kwargs["negative_pooled_prompt_embeds"] = torch.cat(neg_pooled).to(dtype)
    return kwargs
