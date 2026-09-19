# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Evidence checks for the bounded, real SWE1 async SGLang training recipe.

The async trainer logs ``timing/train/weight_sync`` after its blocking refit,
collector weight-version update, and collector resume have returned. These are
completed refit barriers, not independently measured engine weight versions.
"""

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from pathlib import Path


@dataclass(frozen=True)
class TrainingEvidence:
    steps: list[int]
    positive_gradient_steps: list[int]
    completed_refit_steps: list[int]
    active_tokens: dict[int, int]
    rows: dict[int, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _single(row: dict, name: str) -> object:
    value = row.get(name)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must retain the logger's one-row batch dimension")
    return value[0]


def validate_gym_rows(path: Path, expected_rows: int) -> tuple[int, int]:
    """Validate the real logger payload, including effective loss masks."""
    count = active_tokens = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("idx") != count:
                raise ValueError(f"{path}: missing, repeated, or reordered row index")
            agent = _single(row, "agent_ref")
            if not isinstance(agent, dict) or not agent.get("name"):
                raise ValueError(f"{path}: missing Gym agent identity")
            tokens = _single(row, "token_ids")
            mask = _single(row, "token_loss_mask")
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"{path}: empty token IDs")
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError(f"{path}: invalid token ID")
            if not isinstance(mask, list) or len(mask) != len(tokens):
                raise ValueError(f"{path}: token/mask length mismatch")
            sample_mask = _finite(_single(row, "sample_loss_mask"), "sample mask")
            if sample_mask not in (0, 1) or any(value not in (0, 1) for value in mask):
                raise ValueError(f"{path}: non-binary loss mask")
            length = _single(row, "input_lengths")
            if type(length) is not int or not 0 < length <= len(tokens):
                raise ValueError(f"{path}: invalid input length")
            for name in ("generation_logprobs", "prev_logprobs", "advantages"):
                values = _single(row, name)
                if not isinstance(values, list) or len(values) != len(tokens):
                    raise ValueError(f"{path}: {name}/token length mismatch")
                for index, value in enumerate(values):
                    if sample_mask and mask[index]:
                        if index >= length:
                            raise ValueError(f"{path}: active padding token")
                        _finite(value, name)
            _finite(_single(row, "rewards"), "reward")
            active_tokens += int(sample_mask * sum(mask))
            count += 1
    if count != expected_rows or not active_tokens:
        raise ValueError(f"{path}: expected {expected_rows} rows with active tokens")
    return count, active_tokens


def validate_training(
    metrics: dict[str, dict[str, float]],
    log_dir: Path,
    run_log: str,
    returncode: int,
) -> TrainingEvidence:
    """Require useful optimizer work and refits, never just process success."""
    if returncode != 0:
        raise ValueError(f"Training exited with status {returncode}")
    if "Running async GRPO training" not in run_log:
        raise ValueError("No real async GRPO entrypoint evidence")
    expected = {"1", "2", "3"}
    values = {}
    for tag in ("train/loss", "train/grad_norm", "train/global_valid_toks"):
        series = metrics.get(tag, {})
        if set(series) != expected:
            raise ValueError(f"{tag}: expected exactly steps 1, 2, 3")
        values[tag] = {int(step): _finite(value, tag) for step, value in series.items()}
    if any(value <= 0 for value in values["train/global_valid_toks"].values()):
        raise ValueError("Every completed step must have positive valid tokens")
    gradients = values["train/grad_norm"]
    if any(value < 0 for value in gradients.values()):
        raise ValueError("Negative gradient norm")
    positive = sorted(step for step, value in gradients.items() if value > 0)
    if not positive:
        raise ValueError("No finite positive gradient norm")
    refits = metrics.get("timing/train/weight_sync", {})
    if not set(refits).issubset(expected):
        raise ValueError("Refit metrics include an unexpected training step")
    completed_refits = sorted(
        int(step) for step, value in refits.items() if _finite(value, "weight_sync") > 0
    )
    if len(completed_refits) < 2:
        raise ValueError("Fewer than two completed refit barriers")
    rows, active_tokens = {}, {}
    for step in (1, 2, 3):
        matches = list(log_dir.rglob(f"train_data_step{step}.jsonl"))
        if len(matches) != 1:
            raise ValueError(f"Expected one fresh Gym payload for step {step}")
        rows[step], active_tokens[step] = validate_gym_rows(matches[0], 16)
    return TrainingEvidence([1, 2, 3], positive, completed_refits, active_tokens, rows)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.fileMode=true",
            "-C",
            str(root),
            *args,
        ],
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


def source_identity(project: Path, *, allow_runtime_patches: bool = False) -> dict:
    """Allow only the two exact HEAD-derived MCore edits made by SGLang setup.

    Preflight requires clean dependencies. At completion, the narrowly scoped
    substitutions in ``sglang/utils/patches.py`` are permitted, not arbitrary
    dependency dirt or source changes. Every gitlink is checked recursively.
    """
    project = project.resolve()
    mcore = Path(
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
    )
    known_files = {
        "megatron/core/inference/contexts/dynamic_context.py",
        "megatron/training/training.py",
    }
    anchor = b'    torch_memory_saver.hook_mode = "torch"\n'
    replacement = b'    # torch_memory_saver.hook_mode = "torch"  # patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n'
    heads, patches = {}, {}

    def visit(root: Path, expected: str) -> None:
        if (
            root.is_symlink()
            or Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root
        ):
            raise ValueError(f"Missing/redirected source repository: {root}")
        if _git(root, "rev-parse", "HEAD") != expected:
            raise ValueError(f"Wrong source HEAD: {root}")
        if _git(root, "ls-files", "--others", "--exclude-standard"):
            raise ValueError(f"Non-ignored untracked source/artifacts in {root}")
        if _git(
            root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--name-only"
        ):
            raise ValueError(f"Staged source changes: {root}")
        if any(
            not row.startswith("H ")
            for row in _git(root, "ls-files", "-v").splitlines()
        ):
            raise ValueError(f"Hidden index flags: {root}")
        heads[str(root.relative_to(project))] = expected
        for record in _git(root, "ls-tree", "-r", "-z", "HEAD").split("\0"):
            if not record:
                continue
            metadata, name = record.split("\t", 1)
            mode, _, object_id = metadata.split()
            if mode == "160000":
                child = root / name
                if child.resolve() != child:
                    raise ValueError(f"Redirected submodule: {child}")
                visit(child, object_id)
        changed = _git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--name-only",
            "-z",
        ).split("\0")
        for name in filter(None, changed):
            path = root / name
            if (
                not allow_runtime_patches
                or root.relative_to(project) != mcore
                or name not in known_files
            ):
                raise ValueError(f"Unexpected source change: {path}")
            original = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{name}"]
            )
            if (
                path.is_symlink()
                or path.resolve() != path
                or not stat.S_ISREG(path.stat().st_mode)
                or path.stat().st_mode & 0o777 != 0o644
            ):
                raise ValueError(f"Changed runtime patch file type/mode: {path}")
            if original.count(anchor) != 1 or path.read_bytes() != original.replace(
                anchor, replacement, 1
            ):
                raise ValueError(f"Unexpected runtime patch bytes: {path}")
            patches[str(path.relative_to(project))] = sha256(path)

    visit(project, _git(project, "rev-parse", "HEAD"))
    gym_pin = _git(project, "rev-parse", "HEAD:3rdparty/Gym-workspace/Gym")
    return {
        "path": str(project.resolve()),
        "commit": _git(project, "rev-parse", "HEAD"),
        "gym_commit": gym_pin,
        "submodule_heads": heads,
        "runtime_patches": patches,
    }


def validate_data_receipt(
    receipt: dict, train: Path, validation: Path, model: Path, revision: str
) -> None:
    if receipt.get("kind") != "swe1-smoke-data-v1":
        raise ValueError("Unexpected SWE1 data receipt kind")
    if not re.fullmatch(r"[0-9a-f]{40}", receipt.get("dataset_revision", "")):
        raise ValueError("Prepared data requires an immutable dataset revision")
    if (
        receipt["model"]["revision"] != revision
        or Path(receipt["model"]["path"]).resolve() != model.resolve()
    ):
        raise ValueError("Prepared data used a different model snapshot")
    for name, path in (("train", train), ("validation", validation)):
        split = receipt["splits"][name]
        if Path(split["path"]).resolve() != path.resolve() or split["sha256"] != sha256(
            path
        ):
            raise ValueError(f"Prepared {name} path/hash mismatch")
        with path.open() as stream:
            count = sum(1 for line in stream if line.strip())
        if count != split["num_rows"] or count < (12 if name == "train" else 1):
            raise ValueError(f"Prepared {name} row count mismatch/insufficient rows")


def component_imports(project: Path) -> dict[str, str]:
    """Check Gym components as well as its core when reusing editable installs."""
    gym_root = project / "3rdparty/Gym-workspace/Gym"
    imports = {}
    # Match the trainer's order: Gym augments sys.path during its import.
    for name, root in (
        ("nemo_rl", project / "nemo_rl"),
        ("nemo_gym", gym_root / "nemo_gym"),
        ("nemo_gym.openai_utils", gym_root / "nemo_gym"),
        ("responses_api_models.sglang_model.app", gym_root),
        ("responses_api_models.vllm_model.app", gym_root),
        ("responses_api_agents.tool_simulation_agent.app", gym_root),
        (
            "resources_servers.single_step_tool_use_with_argument_comparison.app",
            gym_root,
        ),
    ):
        module = import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"{module.__name__} imported outside the current source")
        imports[name] = module.__file__
    return imports


def preflight(project: Path) -> dict:
    """Read source/assets/imports and attach to an existing Ray cluster only."""
    identity = source_identity(project)
    model = Path(os.environ["NRL_MODEL_PATH"]).resolve()
    revision = os.environ["NRL_MODEL_REVISION"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or model.name != revision:
        raise ValueError(
            "Model must be an immutable snapshot directory named by its commit"
        )
    model_files = [model / "config.json", model / "model.safetensors.index.json"]
    index = json.loads(model_files[1].read_text())
    shards = sorted(set(index["weight_map"].values()))
    if any(Path(shard).is_absolute() or ".." in Path(shard).parts for shard in shards):
        raise ValueError("Model index contains a non-snapshot shard path")
    if not shards or any(
        not (model / shard).is_file() or (model / shard).stat().st_size == 0
        for shard in shards
    ):
        raise ValueError("Model snapshot has missing/empty weight shards")
    receipt_path = Path(os.environ["NRL_SWE1_DATA_RECEIPT"])
    receipt = json.loads(receipt_path.read_text())
    validate_data_receipt(
        receipt,
        Path(os.environ["NRL_SWE1_TRAIN_PATH"]),
        Path(os.environ["NRL_SWE1_VALIDATION_PATH"]),
        model,
        revision,
    )
    if receipt["rendering"]["gym_commit"] != identity["gym_commit"]:
        raise ValueError("Prepared data used a different Gym renderer")

    import ray

    imports = component_imports(project)
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    recipe = (
        project
        / "examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.yaml"
    )
    config = load_config(str(recipe))
    required = {
        "grpo.max_num_steps": 3,
        "grpo.num_prompts_per_step": 4,
        "grpo.num_generations_per_prompt": 4,
        "grpo.async_grpo.enabled": True,
        "grpo.async_grpo.in_flight_weight_updates": False,
        "env.should_use_nemo_gym": True,
        "env.should_log_nemo_gym_responses": False,
        "policy.generation.backend": "sglang",
        "policy.generation.sglang_cfg.moe_runner_backend": "triton",
        "policy.generation.sglang_cfg.quantization.scheme": "bf16",
        "policy.generation.sglang_cfg.sglang_server_config.num_gpus": 4,
        "policy.generation.sglang_cfg.sglang_server_config.num_gpus_per_engine": 2,
        "policy.generation.colocated.enabled": False,
        "policy.megatron_cfg.enabled": True,
        "policy.dtensor_cfg.enabled": False,
        "cluster.num_nodes": 2,
        "cluster.gpus_per_node": 4,
        "logger.tensorboard_enabled": True,
        "checkpointing.enabled": False,
    }
    for key, value in required.items():
        if OmegaConf.select(config, key) != value:
            raise ValueError(
                f"Quick recipe contract changed: {key} must equal {value!r}"
            )
    rendering = receipt["rendering"]
    template_hash = hashlib.sha256(
        config.policy.tokenizer.chat_template.encode()
    ).hexdigest()
    if (
        template_hash != rendering["template_sha256"]
        or OmegaConf.to_container(config.policy.tokenizer.chat_template_kwargs)
        != rendering["chat_template_kwargs"]
    ):
        raise ValueError(
            "Prepared data used a different chat template/rendering configuration"
        )
    if (
        rendering["max_new_tokens"] != config.policy.generation.max_new_tokens
        or rendering["max_prompt_tokens"] + rendering["max_new_tokens"]
        > config.policy.max_total_sequence_length
    ):
        raise ValueError("Prepared data token budget differs from the training recipe")
    address = os.environ.get("RAY_ADDRESS", "auto")
    if address == "local":
        raise ValueError("The driver attaches to an existing two-node Ray cluster")
    ray.init(address=address, log_to_driver=False)
    try:
        nodes = ray.nodes()
        gpu_nodes = [
            node
            for node in nodes
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 4
        ]
        available = ray.available_resources().get("GPU", 0)
        if len(gpu_nodes) < 2 or available < 8:
            raise ValueError(
                "Quick recipe requires two available four-GPU nodes (eight GPUs)"
            )
    finally:
        ray.shutdown()
    versions = {}
    for name in (
        "torch",
        "ray",
        "sglang",
        "flashinfer-python",
        "transformers",
        "nemo-gym",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "source": identity,
        "python": sys.executable,
        "controller_versions": versions,
        "imports": imports,
        "worker_venv_root": os.environ.get("NEMO_RL_VENV_DIR"),
        "gym_venv_root": os.environ["NRL_GYM_VENV_DIR"],
        "image_ref": os.environ.get("NRL_IMAGE_REF"),
        "model": {
            "path": str(model),
            "revision": revision,
            "metadata_sha256": {path.name: sha256(path) for path in model_files},
            "shards": shards,
        },
        "data_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256(receipt_path),
        },
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "cluster": {"gpu_nodes": gpu_nodes, "available_gpus": available},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "source-after", "validate"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.project)
        output = args.run_dir / "preflight.json"
    elif args.command == "source-after":
        result = source_identity(args.project, allow_runtime_patches=True)
        output = args.run_dir / "source-after.json"
    else:
        before = json.loads((args.run_dir / "preflight.json").read_text())
        after = source_identity(args.project, allow_runtime_patches=True)
        if before["source"] | {"runtime_patches": after["runtime_patches"]} != after:
            raise ValueError("Source identity changed during training")
        data = before["data_receipt"]
        if sha256(Path(data["path"])) != data["sha256"]:
            raise ValueError("Prepared data receipt changed during training")
        receipt = json.loads(Path(data["path"]).read_text())
        model = before["model"]
        validate_data_receipt(
            receipt,
            Path(os.environ["NRL_SWE1_TRAIN_PATH"]),
            Path(os.environ["NRL_SWE1_VALIDATION_PATH"]),
            Path(model["path"]),
            model["revision"],
        )
        if any(
            sha256(Path(model["path"]) / name) != expected
            for name, expected in model["metadata_sha256"].items()
        ):
            raise ValueError("Model metadata changed during training")
        evidence = validate_training(
            json.loads((args.run_dir / "metrics.json").read_text()),
            args.run_dir / "logs",
            (args.run_dir / "run.log").read_text(),
            int((args.run_dir / "training.exitcode").read_text()),
        )
        result = {"source": after, "training": asdict(evidence)}
        output = args.run_dir / "validation.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
