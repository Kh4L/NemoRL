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

"""CPU negative controls for real SWE1 acceptance evidence; no training stubs."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tests.swe1_sglang_checks import (
    sha256,
    source_identity,
    validate_data_receipt,
    validate_gym_rows,
    validate_training,
)


def gym_row(index: int) -> dict:
    # Logger.log_batched_dict_as_jsonl keeps a size-one leading batch dimension.
    return {
        "idx": index,
        "agent_ref": [{"name": "single_step_tool_use_with_argument_comparison_swe"}],
        "token_ids": [[10, 20, 30, 0]],
        "token_loss_mask": [[0.0, 1.0, 1.0, 0.0]],
        "sample_loss_mask": [1.0],
        "input_lengths": [3],
        "generation_logprobs": [[0.0, -0.5, -0.25, 0.0]],
        "prev_logprobs": [[0.0, -0.6, -0.3, 0.0]],
        "advantages": [[0.0, 0.5, 0.5, 0.0]],
        "rewards": [1.0],
    }


@pytest.fixture
def evidence(tmp_path: Path) -> dict:
    for step in (1, 2, 3):
        (tmp_path / f"train_data_step{step}.jsonl").write_text(
            "".join(json.dumps(gym_row(index)) + "\n" for index in range(16))
        )
    return {
        "metrics": {
            "train/loss": {"1": 0.1, "2": 0.0, "3": -0.2},
            "train/grad_norm": {"1": 0.0, "2": 0.5, "3": 0.0},
            "train/global_valid_toks": {"1": 32.0, "2": 32.0, "3": 32.0},
            "timing/train/weight_sync": {"1": 0.2, "2": 0.3, "3": 0.4},
        },
        "log_dir": tmp_path,
        "run_log": "🚀 Running async GRPO training\n",
        "returncode": 0,
    }


def test_accepts_useful_training_without_requiring_every_gradient_nonzero(evidence):
    result = validate_training(**evidence)
    assert result.steps == [1, 2, 3]
    assert result.positive_gradient_steps == [2]
    assert result.completed_refit_steps == [1, 2, 3]
    assert result.active_tokens == {1: 32, 2: 32, 3: 32}


@pytest.mark.parametrize(
    ("tag", "series", "message"),
    [
        ("train/loss", {"3": 0.0}, "exactly steps"),
        ("train/loss", {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}, "exactly steps"),
        ("train/loss", {"1": float("nan"), "2": 0.0, "3": 0.0}, "finite"),
        ("train/grad_norm", {"1": 0.0, "2": 0.0, "3": 0.0}, "positive gradient"),
        ("train/grad_norm", {"1": 1.0, "2": -1.0, "3": 0.0}, "Negative gradient"),
        (
            "train/global_valid_toks",
            {"1": 1.0, "2": 0.0, "3": 1.0},
            "positive valid tokens",
        ),
        ("timing/train/weight_sync", {"1": 1.0}, "two completed refit"),
        (
            "timing/train/weight_sync",
            {"1": 0.0, "2": 0.0, "3": 0.0},
            "two completed refit",
        ),
        ("timing/train/weight_sync", {"1": 1.0, "2": float("inf")}, "finite"),
        (
            "timing/train/weight_sync",
            {"0": 1.0, "1": 1.0, "2": 1.0},
            "unexpected training step",
        ),
    ],
)
def test_rejects_incomplete_or_useless_metrics(evidence, tag, series, message):
    evidence["metrics"][tag] = series
    with pytest.raises(ValueError, match=message):
        validate_training(**evidence)


@pytest.mark.parametrize(
    ("field", "value"), [("returncode", 1), ("run_log", "fake success")]
)
def test_rejects_failed_or_non_async_driver(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_training(**evidence)


def test_rejects_ambiguous_reused_gym_artifacts(evidence):
    duplicate = evidence["log_dir"] / "previous"
    duplicate.mkdir()
    (duplicate / "train_data_step1.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="one fresh Gym payload"):
        validate_training(**evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("idx", 1, "row index"),
        ("agent_ref", [{}], "Gym agent"),
        ("token_ids", [[10, 20]], "length mismatch"),
        ("token_ids", [[10, -1, 30, 0]], "invalid token"),
        ("token_loss_mask", [[0, 0, 0, 0]], "active tokens"),
        ("token_loss_mask", [[0, 1, 1, 1]], "active padding"),
        ("sample_loss_mask", [0], "active tokens"),
        ("sample_loss_mask", [0.5], "non-binary"),
        ("generation_logprobs", [[0, float("nan"), 0, 0]], "finite"),
        ("prev_logprobs", [[0, -0.5]], "length mismatch"),
        ("advantages", [[0, float("inf"), 0, 0]], "finite"),
        ("input_lengths", [0], "input length"),
        ("rewards", [float("nan")], "finite"),
        ("token_loss_mask", [0, 1, 1, 0], "batch dimension"),
    ],
)
def test_rejects_invalid_gym_payload(tmp_path, field, value, message):
    row = gym_row(0)
    row[field] = value
    path = tmp_path / "payload.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        validate_gym_rows(path, 1)


def test_inactive_logprobs_do_not_count_as_training_evidence(tmp_path):
    row = gym_row(0)
    row["generation_logprobs"][0][0] = float("nan")
    path = tmp_path / "payload.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert validate_gym_rows(path, 1) == (1, 2)


@pytest.fixture
def prepared_data(tmp_path):
    paths = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
    }
    for path in paths.values():
        path.write_text("{}\n" * 12)
    model = tmp_path / ("a" * 40)
    receipt = {
        "kind": "swe1-smoke-data-v1",
        "dataset_revision": "b" * 40,
        "model": {"path": str(model), "revision": "a" * 40},
        "splits": {
            name: {"path": str(path), "sha256": sha256(path), "num_rows": 12}
            for name, path in paths.items()
        },
    }
    return receipt, paths["train"], paths["validation"], model, "a" * 40


def test_prepared_data_exact_paths_hashes_and_counts(prepared_data):
    validate_data_receipt(*prepared_data)


@pytest.mark.parametrize(
    "change", ["model", "path", "hash", "count", "kind", "revision"]
)
def test_rejects_changed_prepared_data(prepared_data, change):
    receipt, *args = prepared_data
    receipt = copy.deepcopy(receipt)
    if change == "model":
        receipt["model"]["revision"] = "b" * 40
    elif change == "kind":
        receipt["kind"] = "other"
    elif change == "revision":
        receipt["dataset_revision"] = "main"
    else:
        field = {"path": "path", "hash": "sha256", "count": "num_rows"}[change]
        receipt["splits"]["train"][field] = 1 if change == "count" else "wrong"
    with pytest.raises(ValueError):
        validate_data_receipt(receipt, *args)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def init_repo(path):
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "source.py").write_text("# source\n")
    git(path, "add", "source.py")
    git(
        path,
        "-c",
        "user.name=Serge Panev",
        "-c",
        "user.email=spanev@nvidia.com",
        "commit",
        "-s",
        "-qm",
        "test: create source fixture",
    )
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def source_tree(tmp_path):
    project = tmp_path / "source"
    init_repo(project)
    gym = project / "3rdparty/Gym-workspace/Gym"
    gym_sha = init_repo(gym)
    mcore = (
        project
        / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
    )
    init_repo(mcore)
    for name in (
        "megatron/core/inference/contexts/dynamic_context.py",
        "megatron/training/training.py",
    ):
        path = mcore / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('if True:\n    torch_memory_saver.hook_mode = "torch"\n')
    git(mcore, "add", "megatron")
    git(
        mcore,
        "-c",
        "user.name=Serge Panev",
        "-c",
        "user.email=spanev@nvidia.com",
        "commit",
        "-s",
        "-qm",
        "test: create MCore fixture",
    )
    for path, sha in ((gym, gym_sha), (mcore, git(mcore, "rev-parse", "HEAD"))):
        git(
            project,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{sha},{path.relative_to(project)}",
        )
    git(
        project,
        "-c",
        "user.name=Serge Panev",
        "-c",
        "user.email=spanev@nvidia.com",
        "commit",
        "-s",
        "-qm",
        "test: pin source fixtures",
    )
    return project, mcore


def test_source_requires_clean_preflight_and_only_exact_runtime_patch(source_tree):
    project, mcore = source_tree
    before = source_identity(project)
    assert before["runtime_patches"] == {}
    path = mcore / "megatron/training/training.py"
    path.write_text(
        path.read_text().replace(
            '    torch_memory_saver.hook_mode = "torch"\n',
            '    # torch_memory_saver.hook_mode = "torch"  # patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n',
        )
    )
    with pytest.raises(ValueError, match="Unexpected source change"):
        source_identity(project)
    after = source_identity(project, allow_runtime_patches=True)
    assert after["commit"] == before["commit"]
    assert len(after["runtime_patches"]) == 1
    path.write_text(path.read_text() + "# extra change\n")
    with pytest.raises(ValueError, match="Unexpected runtime patch bytes"):
        source_identity(project, allow_runtime_patches=True)


@pytest.mark.parametrize(
    "change", ["main", "gym", "dependency", "staged", "hidden", "untracked"]
)
def test_source_rejects_other_dirt(source_tree, change):
    project, mcore = source_tree
    root = (
        project / "3rdparty/Gym-workspace/Gym"
        if change == "gym"
        else mcore
        if change == "dependency"
        else project
    )
    (root / "source.py").write_text("# unexpected change\n")
    if change == "staged":
        git(root, "add", "source.py")
    elif change == "hidden":
        git(root, "update-index", "--assume-unchanged", "source.py")
    elif change == "untracked":
        # Restore the tracked fixture so only the extra importable file differs.
        (root / "source.py").write_text("# source\n")
        (root / "untracked_module.py").write_text("# must not affect imports\n")
    with pytest.raises(ValueError):
        source_identity(project, allow_runtime_patches=True)


def test_source_rejects_changed_submodule_head(source_tree):
    project, mcore = source_tree
    git(
        mcore,
        "-c",
        "user.name=Serge Panev",
        "-c",
        "user.email=spanev@nvidia.com",
        "commit",
        "-s",
        "--allow-empty",
        "-qm",
        "test: change dependency fixture",
    )
    with pytest.raises(ValueError, match="Wrong source HEAD"):
        source_identity(project, allow_runtime_patches=True)


@pytest.mark.parametrize("change", ["mode", "symlink", "staged_patch"])
def test_source_rejects_runtime_patch_metadata_changes(source_tree, change):
    project, mcore = source_tree
    path = mcore / "megatron/training/training.py"
    if change == "mode":
        path.chmod(0o755)
    elif change == "symlink":
        original = path.read_text()
        path.unlink()
        target = mcore / "copy.py"
        target.write_text(original)
        path.symlink_to(target)
    else:
        path.write_text(path.read_text() + "# modified\n")
        git(mcore, "add", str(path.relative_to(mcore)))
    with pytest.raises(ValueError):
        source_identity(project, allow_runtime_patches=True)
