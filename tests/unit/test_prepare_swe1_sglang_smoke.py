"""Selection invariants; real tokenizer/Gym validation is a separate preflight."""

import json
from pathlib import Path

import pytest

from tools.prepare_swe1_sglang_smoke import (
    file_sha256,
    select_rows,
    validate_template_kwargs,
)


def test_template_kwargs_preserves_valid_options_without_mutation():
    options = {"enable_thinking": True, "truncate_history_thinking": False}
    actual = validate_template_kwargs(options)
    assert actual == options
    assert actual is not options
    assert validate_template_kwargs({}) == {}


@pytest.mark.parametrize("value", [None, [], "enable_thinking", {1: True}])
def test_template_kwargs_rejects_non_mapping_or_non_string_keys(value):
    with pytest.raises(ValueError, match="string-keyed mapping"):
        validate_template_kwargs(value)


def write_rows(path: Path, prompt_lengths: list[int]) -> list[bytes]:
    rows = [
        (
            json.dumps(
                {
                    "responses_create_params": {"input": str(length)},
                    "expected_action": {
                        "type": "function_call",
                        "name": "test",
                        "arguments": "{}",
                    },
                    "agent_ref": {"type": "responses_api_agents", "name": "test_agent"},
                    "metadata": {"index": index},
                }
            )
            + "\n"
        ).encode()
        for index, length in enumerate(prompt_lengths)
    ]
    path.write_bytes(b"".join(rows))
    return rows


def test_selection_preserves_whole_rows_and_never_renders_labels(tmp_path: Path):
    path = tmp_path / "source.jsonl"
    raw_rows = write_rows(path, [8, 3, 4, 5])
    original_hash = file_sha256(path)
    observed = []

    def render(params):
        observed.append(params)
        assert set(params) == {"input"}
        return [7] * int(params["input"])

    selected, counters = select_rows(
        path, count=2, max_prompt_tokens=4, prompt_token_ids=render
    )
    assert selected == [(2, raw_rows[1], 3), (3, raw_rows[2], 4)]
    assert counters == {"examined": 3, "over_budget": 1, "duplicate_prompt": 0}
    assert len(observed) == 3
    assert file_sha256(path) == original_hash


def test_duplicate_prompts_are_not_split_into_train_and_validation(tmp_path: Path):
    path = tmp_path / "source.jsonl"
    raw_rows = write_rows(path, [2, 2, 3])
    selected, counters = select_rows(
        path,
        count=2,
        max_prompt_tokens=4,
        prompt_token_ids=lambda params: [1] * int(params["input"]),
    )
    assert selected == [(1, raw_rows[0], 2), (3, raw_rows[2], 3)]
    assert counters["duplicate_prompt"] == 1


def test_insufficient_short_rows_fail_without_truncation(tmp_path: Path):
    path = tmp_path / "source.jsonl"
    write_rows(path, [5, 8])
    with pytest.raises(ValueError, match="Do not truncate"):
        select_rows(
            path, count=1, max_prompt_tokens=4, prompt_token_ids=lambda _: [1] * 5
        )


def test_missing_routing_or_label_fails_before_rendering(tmp_path: Path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"responses_create_params": {"input": "hello"}}) + "\n")

    def render(_):
        pytest.fail("An invalid row must not be rendered")

    with pytest.raises(ValueError, match="expected action or agent routing"):
        select_rows(path, count=1, max_prompt_tokens=4, prompt_token_ids=render)


def test_empty_rendered_prompt_is_an_error(tmp_path: Path):
    path = tmp_path / "source.jsonl"
    write_rows(path, [3])
    with pytest.raises(ValueError, match="empty rendered prompt"):
        select_rows(path, count=1, max_prompt_tokens=4, prompt_token_ids=lambda _: [])
