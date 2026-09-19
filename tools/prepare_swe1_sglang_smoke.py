# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select unmodified short SWE1 pivots for integration testing, not evaluation."""

import argparse
import hashlib
import inspect
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@dataclass
class SmokeDataSplit:
    """Provenance for one unmodified subset of the source rows."""

    path: str
    sha256: str
    num_rows: int
    source_lines: list[int]
    input_token_counts: list[int]


def validate_template_kwargs(value: object) -> dict[str, Any]:
    """Require a keyword mapping without coercing or dropping invalid keys."""
    if not isinstance(value, dict):
        raise ValueError("chat_template_kwargs must be a string-keyed mapping")
    result: dict[str, Any] = {}
    for key, option in value.items():
        if not isinstance(key, str):
            raise ValueError("chat_template_kwargs must be a string-keyed mapping")
        result[key] = option
    return result


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_rows(
    source: Path,
    *,
    count: int,
    max_prompt_tokens: int,
    prompt_token_ids: Callable[[dict[str, Any]], list[int]],
) -> tuple[list[tuple[int, bytes, int]], dict[str, int]]:
    """Keep complete unique rows in source order, filtering only by prompt length."""
    selected: list[tuple[int, bytes, int]] = []
    seen: set[str] = set()
    counters = {"examined": 0, "over_budget": 0, "duplicate_prompt": 0}
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not row.get("expected_action") or not row.get("agent_ref"):
                raise ValueError(
                    f"Line {line_number} lacks an expected action or agent routing"
                )
            prompt_key = json.dumps(row["responses_create_params"], sort_keys=True)
            counters["examined"] += 1
            if prompt_key in seen:
                counters["duplicate_prompt"] += 1
                continue
            seen.add(prompt_key)
            token_count = len(prompt_token_ids(row["responses_create_params"]))
            if token_count > max_prompt_tokens:
                counters["over_budget"] += 1
                continue
            if token_count == 0:
                raise ValueError(f"Line {line_number} has an empty rendered prompt")
            selected.append((line_number, raw_line, token_count))
            if len(selected) == count:
                return selected, counters
    raise ValueError(
        f"Only {len(selected)} unique prompts fit {max_prompt_tokens} tokens; "
        f"need {count}. Do not truncate pivots to force a passing smoke test."
    )


def main() -> None:
    """Use the actual native adapter renderer without making a model request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.yaml"
        ),
    )
    args = parser.parse_args()
    if args.train_count <= 0 or args.validation_count <= 0:
        parser.error("Both split sizes must be positive")
    for revision in (args.model_revision, args.dataset_revision):
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            parser.error("Revisions must be full lowercase commit hashes")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new evidence directory")
    if args.model_path.name != args.model_revision:
        parser.error("Model path must name the immutable revision directory")

    # Gym is optional outside the preparation command; keep lightweight selection
    # tests independent of the model-server environment.
    import nemo_gym
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import BaseServerConfig, ServerClient
    from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

    gym_root = Path(nemo_gym.__file__).resolve().parent.parent
    expected_gym = Path(__file__).resolve().parents[1] / "3rdparty/Gym-workspace/Gym"
    if gym_root != expected_gym.resolve() or any(
        not Path(inspect.getfile(model_class)).resolve().is_relative_to(gym_root)
        for model_class in (SGLangModel, SGLangModelConfig, VLLMModel, VLLMModelConfig)
    ):
        raise RuntimeError(
            "Gym core and model components must come from the pinned submodule; "
            "set NEMO_GYM_EXTRA_ROOTS to that checkout before starting Python"
        )

    register_omegaconf_resolvers()
    config = load_config(args.config)
    template = config.policy.tokenizer.chat_template
    template_kwargs = validate_template_kwargs(
        OmegaConf.to_container(
            config.policy.tokenizer.chat_template_kwargs, resolve=True
        )
    )
    context_length = config.policy.max_total_sequence_length
    max_new_tokens = config.policy.generation.max_new_tokens
    prompt_budget = context_length - max_new_tokens
    if prompt_budget <= 0:
        parser.error("The generation budget leaves no room for a prompt")
    model_path = args.model_path.resolve()
    model = SGLangModel(
        config=SGLangModelConfig(
            name="policy_model",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            model=str(model_path),
            base_url="http://127.0.0.1:1/v1",
            api_key="unused",
            context_length=context_length,
            return_token_id_information=True,
            uses_reasoning_parser=True,
            sglang_tool_format="hermes",
            sglang_chat_template=template,
            chat_template_kwargs=template_kwargs,
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create({}),
        ),
    )

    def prompt_token_ids(params: dict[str, Any]) -> list[int]:
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
        chat_body = model.get_converter().responses_to_chat_completion_create_params(
            body
        )
        prepared = model._prepare_sglang_prompt(
            chat_body.model_dump(exclude_unset=True)
        )
        return model._full_sglang_tokenize(*prepared)

    selected, counters = select_rows(
        args.source,
        count=args.train_count + args.validation_count,
        max_prompt_tokens=prompt_budget,
        prompt_token_ids=prompt_token_ids,
    )
    gym_commit = subprocess.check_output(
        ["git", "-C", str(gym_root), "rev-parse", "HEAD"], text=True
    ).strip()
    splits: dict[str, dict[str, Any]] = {}
    receipt = {
        "kind": "swe1-smoke-data-v1",
        "scope": "integration-only; original rows preserved; no quality/convergence claim",
        "source": {
            "path": str(args.source.resolve()),
            "sha256": file_sha256(args.source),
        },
        "dataset_revision": args.dataset_revision,
        "model": {"path": str(model_path), "revision": args.model_revision},
        "rendering": {
            "gym_commit": gym_commit,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "chat_template_kwargs": template_kwargs,
            "max_prompt_tokens": prompt_budget,
            "max_new_tokens": max_new_tokens,
        },
        "selection": counters,
        "splits": splits,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in (
        ("train", selected[: args.train_count]),
        ("validation", selected[args.train_count :]),
    ):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("xb") as stream:
            for _, raw_line, _ in rows:
                stream.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        splits[split] = asdict(
            SmokeDataSplit(
                path=str(path.resolve()),
                sha256=file_sha256(path),
                num_rows=len(rows),
                source_lines=[row[0] for row in rows],
                input_token_counts=[row[2] for row in rows],
            )
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
