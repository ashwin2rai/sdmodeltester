"""Prompt processing: one-level dynamic alternatives and (later) weighted embeddings.

Dynamic syntax::

    a man with {white | black | silver} hair
    a man with {(white:1.2) | black} hair

Rules (SPEC §9.1): one brace level only, at least two non-empty alternatives
per group, alternatives are whitespace-trimmed, several groups per prompt are
allowed. Choices are made with a PRNG derived from the image seed (SPEC §9.2),
never with the diffusion generator.

The weighted-embedding adapter (``(text:1.2)`` -> prompt_embeds) is added in
Phase 3 and must stay isolated in this module.
"""

from __future__ import annotations

import random

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
