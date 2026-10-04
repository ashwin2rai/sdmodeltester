import random

import pytest

from src.prompting import (
    PromptSyntaxError,
    has_dynamic_groups,
    parse_template,
    resolve_prompt_pair,
    resolve_template,
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
