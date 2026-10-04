import random

import pytest

from src.prompting import (
    CHUNK_TOKENS,
    PromptSyntaxError,
    chunk_count,
    chunk_tokens,
    has_dynamic_groups,
    parse_template,
    parse_weighted,
    resolve_prompt_pair,
    resolve_template,
    weighted_token_ids,
)


def test_plain_prompt_is_unchanged():
    assert resolve_template("a cat, (fur:1.2)", random.Random(0)) == "a cat, (fur:1.2)"
    assert not has_dynamic_groups("a cat")


def test_alternatives_are_trimmed():
    assert parse_template("a { white |black } cat") == ["a ", ("white", "black"), " cat"]


def test_count_ten_produces_only_valid_alternatives():
    results = {resolve_prompt_pair("a {white | black} cat", "", seed)[0] for seed in range(10)}
    assert results <= {"a white cat", "a black cat"}


def test_fixed_seed_reproduces_choice():
    template = "a {red | green | blue | white} {cat | dog} in {rain | snow}"
    assert resolve_prompt_pair(template, "", 123) == resolve_prompt_pair(template, "", 123)


def test_different_seeds_can_differ():
    template = "a {red | green | blue | white} {cat | dog}"
    results = {resolve_prompt_pair(template, "", seed)[0] for seed in range(50)}
    assert len(results) > 1


def test_weight_syntax_preserved_inside_alternatives():
    results = {resolve_prompt_pair("a {(white:1.2) | black} cat", "", s)[0] for s in range(50)}
    assert results == {"a (white:1.2) cat", "a black cat"}


def test_negative_prompt_supports_groups():
    _, negative = resolve_prompt_pair("cat", "{blurry | lowres}", 7)
    assert negative in {"blurry", "lowres"}


def test_multiple_groups_allowed():
    segments = parse_template("{a | b} and {c | d | e}")
    assert segments == [("a", "b"), " and ", ("c", "d", "e")]


@pytest.mark.parametrize(
    "template",
    [
        "a {red | {blue | green}} cat",
        "a {red | } cat",
        "a {} cat",
        "a {red} cat",
        "a {red | blue cat",
        "a red | blue} cat",
    ],
)
def test_invalid_templates_rejected(template):
    with pytest.raises(PromptSyntaxError):
        parse_template(template)


# --- weighted prompts (pure Python parts) -----------------------------------------


def test_parse_weighted_basic():
    assert parse_weighted("a (white hair:1.2) cat") == [
        ("a ", 1.0),
        ("white hair", 1.2),
        (" cat", 1.0),
    ]


def test_parse_weighted_variants():
    assert parse_weighted("(bg:0.7)") == [("bg", 0.7)]
    assert parse_weighted("(bg: .5 )") == [("bg", 0.5)]
    assert parse_weighted("(x:2)(y:2)") == [("xy", 2.0)]  # equal weights merge
    assert parse_weighted("") == []


def test_parse_weighted_literal_parentheses():
    # Only explicit numeric weights are syntax; everything else stays literal.
    assert parse_weighted("a (plain) cat") == [("a (plain) cat", 1.0)]
    assert parse_weighted("a (ratio:16/9) b") == [("a (ratio:16/9) b", 1.0)]
    # The last ":number)" wins, as in A1111: escape parentheses to keep them literal.
    assert parse_weighted("(ratio:16:9)") == [("ratio:16", 9.0)]
    assert parse_weighted(r"\(x:1.5\)") == [("(x:1.5)", 1.0)]
    assert parse_weighted(r"(a \(b\):1.3)") == [("a (b)", 1.3)]


def test_weighted_token_ids_assigns_weights():
    def tokenize(text):  # one token per non-space character
        return [ord(c) for c in text if c != " "]

    ids, weights = weighted_token_ids("ab (cd:1.5) e", tokenize)
    assert ids == [ord(c) for c in "abcde"]
    assert weights == [1.0, 1.0, 1.5, 1.5, 1.0]


def test_chunk_tokens_single_window():
    chunks = chunk_tokens([5, 6], [1.0, 2.0], bos=0, eos=1, pad=9)
    assert len(chunks) == 1
    ids, weights = chunks[0]
    assert len(ids) == len(weights) == CHUNK_TOKENS + 2
    assert ids[:4] == [0, 5, 6, 1] and set(ids[4:]) == {9}
    assert weights[:4] == [1.0, 1.0, 2.0, 1.0]


def test_chunk_tokens_long_prompt_and_padding_to_common_length():
    ids = list(range(100, 100 + CHUNK_TOKENS + 5))
    chunks = chunk_tokens(ids, [1.0] * len(ids), bos=0, eos=1, pad=1)
    assert len(chunks) == 2
    assert chunks[1][0][:7] == [0, *ids[CHUNK_TOKENS:], 1]
    assert len(chunk_tokens([], [], bos=0, eos=1, pad=1)) == 1
    padded = chunk_tokens([7], [1.0], bos=0, eos=1, pad=1, num_chunks=3)
    assert len(padded) == 3 and padded[2][0][:2] == [0, 1]
    with pytest.raises(ValueError):
        chunk_tokens(ids, [1.0] * len(ids), bos=0, eos=1, pad=1, num_chunks=1)


def test_chunk_count():
    assert [chunk_count(n) for n in (0, 1, 75, 76, 150, 151)] == [1, 1, 1, 2, 2, 3]
