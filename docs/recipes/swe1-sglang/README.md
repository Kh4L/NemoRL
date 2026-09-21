# SWE1 with SGLang: a three-step training recipe

Native NeMo-Gym generation with SGLang and asynchronous GRPO for
Qwen3-30B-A3B-Thinking-2507. Megatron trains the policy; SGLang generates
responses. Collection is asynchronous, while weight updates use blocking refit
barriers.

**Publication candidate:** the instructions below require the pinned source
commits to be published in the designated public forks. Their inclusion here
does not mean those commits are already available or approved for release.

The bounded recipe completed three training steps on **two GB200 nodes with
four GPUs each**, with two positive gradient norms and three training-time
refits. See [VALIDATION.md](VALIDATION.md) for the measured result and limits.
The run reused compiled environments, including a retained Transformer Engine
wheel: **a fresh environment build has not been demonstrated**.

This guide does not provide model weights, datasets, compiled environments or
a cluster launcher. Run preparation and training in compute allocations, not
on a shared login host.

## Fixed integration profile

| Setting | Value |
| --- | --- |
| Training | 4 GPUs; TP2 / PP1 / CP1 / EP4 / expert TP1 |
| Generation | 4 GPUs; two TP2 SGLang engines |
| Steps and batch | 3 steps; 4 prompts × 4 generations = 16 rows per step |
| Token budgets | 22,528 total / 16,384 new / 6,144 prompt tokens |
| Precision and MoE runner | BF16 and `triton` |
| Checkpoint saving | Disabled |

Use this configuration inside the restored NeMo-RL checkout:

```text
examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.yaml
```

Its relative defaults require the original source layout. Do not substitute
the inherited 128-GPU configuration or run a copied YAML from another directory.

## 1. Restore the pinned source

After publication, use Bash, Git, a SHA-256 utility and network access to the
designated public repositories. First obtain this guide without initializing
its submodules; the bootstrap then creates a separate, pinned runtime checkout:

```bash
git clone --no-recurse-submodules --single-branch \
  --branch recipe/swe1-sglang-20260921 \
  https://github.com/Kh4L/NemoRL.git /absolute/new/recipe-guide
cd /absolute/new/recipe-guide/docs/recipes/swe1-sglang
sha256sum -c SHA256SUMS  # macOS: shasum -a 256 -c SHA256SUMS

# Choose an absolute destination that does not already exist.
bash scripts/checkout_sources.sh /absolute/new/NeMo-RL docs
cd /absolute/new/NeMo-RL
```

The source locations designated for publication are `https://github.com/Kh4L/NemoRL.git`
and `https://github.com/Kh4L/NemoGym.git`. The bootstrap fetches the exact NeMo
commit, verifies its Gym gitlink against the lock, and sets that checkout's Gym
submodule URL to the designated fork before recursive initialization. It does
not edit `.gitmodules` or advance submodule pins. It rejects an existing or
symlink destination and does not install dependencies or launch a job. Retain
a partial checkout for diagnosis on failure.

The `docs` profile selects NeMo-RL `9271c3a9718127d02c1c30ede3557c6aee3d95ec`.
It differs from the tested runtime only in `docs/guides/swe-rl-qwen3.md`.
Use `runtime` instead of `docs` for the exact tested commit
`e4df35ba3ab3c20906335fa7ff978b9766778179`. Both require Gym
`ea5a37ea8b78a986cdb2e670b14012cd3981df03` and the pinned recursive submodules.
Do not replace these pins with the current tips of the forks.

## 2. Prepare the environments

Use `uv` and the restored source's `pyproject.toml`, `uv.lock` and
`docs/docker.md`. Prepare separate environments for the `mcore`, `sglang`,
`vllm` and `nemo_gym` extras; their dependencies conflict. The async collector
and replay buffer need the **vLLM environment even though SGLang generates the
responses**. Prepare a controller with the Gym dependencies as well.

Compilation, system libraries, model/data downloads and GPU-driver setup are
separate prerequisites. Do not silently advance dependency pins or substitute
a stock Gym checkout without the native `sglang_model` adapter.

`NEMO_RL_VENV_DIR` is a parent directory, not a shared virtual environment.
Each entry below must resolve to the appropriate environment with `bin/python`:

| Directory under `NEMO_RL_VENV_DIR` | Environment |
| --- | --- |
| `nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker` | `mcore` |
| `nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker` | `sglang` |
| `nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector` | `vllm` |
| `nemo_rl.algorithms.async_utils.ReplayBuffer` | `vllm` |
| `nemo_rl.environments.nemo_gym.NemoGym` | `nemo_gym` |
| `nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor` | `nemo_gym` |

Gym services use a different layout under `NRL_GYM_VENV_DIR`. Each needs both
`.venv/bin/python` and `.venv/bin/activate`:

```text
responses_api_models/sglang_model/.venv/
responses_api_agents/tool_simulation_agent/.venv/
resources_servers/single_step_tool_use_with_argument_comparison/.venv/
```

On both allocated nodes, verify actual interpreter prefixes, package versions,
module origins and compiled imports before starting Ray. Reused editable
installs must resolve to the selected checkout. Megatron-Bridge's Python source
is under `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src`, not the
repository root. Do not add one role's entire `site-packages` to every role's
`PYTHONPATH`.

## 3. Prepare the small data subset

Obtain the pinned model snapshot and the **placeholder-filled SWE1 training
split** from `nvidia/Nemotron-RL-Super-Training-Blends`. Raw downloaded JSONL is
not the prepared input. Follow the restored
`docs/guides/nemotron-3-super.md` for placeholder filling and the original split.
Model and dataset access and their terms are separate from this source recipe.

From the restored NeMo-RL checkout, replace every `/path/to/...` value:

```bash
unset NEMO_RL_PY_EXECUTABLES_SYSTEM LD_PRELOAD
export NRL_MODEL_REVISION=144afc2f379b542fdd4e85a1fcd5e1f79112d95d
export NRL_MODEL_PATH=/path/to/model/snapshots/$NRL_MODEL_REVISION
export NRL_DATASET_REVISION=b90f74f1d0bafeec6d1f1321173f6775ba5bda2e
export NRL_SWE1_SOURCE=/path/to/prepared/swe1/train-split.jsonl
export NRL_PREPARED_DATA=/path/to/new/swe1-smoke-data
export NEMO_RL_VENV_DIR=/path/to/verified/actor-venvs
export NRL_GYM_VENV_DIR=/path/to/verified/gym-service-venvs
export NRL_RUN_DIR=/path/to/new/swe1-smoke-run
export NRL_MEGATRON_CHECKPOINT_DIR=/path/to/writable/model-conversion-cache
export NEMO_GYM_EXTRA_ROOTS="$PWD/3rdparty/Gym-workspace/Gym"

uv run --frozen --extra nemo_gym tools/prepare_swe1_sglang_smoke.py \
  --source "$NRL_SWE1_SOURCE" --output-dir "$NRL_PREPARED_DATA" \
  --model-path "$NRL_MODEL_PATH" --model-revision "$NRL_MODEL_REVISION" \
  --dataset-revision "$NRL_DATASET_REVISION"

export NRL_SWE1_TRAIN_PATH=$NRL_PREPARED_DATA/train.jsonl
export NRL_SWE1_VALIDATION_PATH=$NRL_PREPARED_DATA/validation.jsonl
export NRL_SWE1_DATA_RECEIPT=$NRL_PREPARED_DATA/manifest.json
```

The model directory must end in its full revision hash. The conversion-cache
directory must already exist and be writable. `NRL_RUN_DIR` must be absolute and
not yet exist. Keep generated data and run output outside the source checkout.

The preparation tool selects 12 training and four validation rows using prompt
length and deduplication, not rewards. It preserves complete prompts, labels and
routing and fails if too few fit the 6,144-token prompt budget. Do not truncate
examples to force acceptance. Both subsets come from the **training split**;
they are not a held-out benchmark. Retain the preparation manifest.

## 4. Set the runtime environment on both nodes

Apply the same reviewed settings to the controller and both node launch
environments before starting Ray. For the tested CUDA 13 layout:

```bash
unset NEMO_RL_PY_EXECUTABLES_SYSTEM LD_PRELOAD
export NRL_FORCE_REBUILD_VENVS=false
export NRL_SHARED_CUDA_SITE=/path/to/verified/cuda/site-packages
export CUDA_HOME=/path/to/verified/cuda-toolkit
export TORCH_CUDA_ARCH_LIST="10.0"  # GB200; match the actual target GPU.
export LD_LIBRARY_PATH="$NRL_SHARED_CUDA_SITE/nvidia/cu13/lib:$NRL_SHARED_CUDA_SITE/nvidia/nvshmem/lib"
```

- Megatron needs `TORCH_CUDA_ARCH_LIST` in the driver runtime, even with
  precompiled packages. A different GPU or CUDA layout requires its own checks.
- Verify selected library versions and hashes against every role. Do not append
  an inherited loader path, driver stubs or another role's `nvidia/cudnn/lib`.
  cuDNN Frontend can explicitly open that copy after PyTorch loads a role-local
  copy, producing duplicate providers.
- Measure loaded libraries after real role imports on both nodes and reject
  duplicate providers. Matching package versions alone are insufficient. The
  acceptance driver below does not perform this measurement.
- Retain SGLang Miles revision `3003d70f680d41c59d1b7acbf65cb47795dfd19e`,
  BF16 and the `triton` MoE runner. Do not silently switch to the BF16
  FlashInfer TRT-LLM expert-reload path.

## 5. Start Ray and run the acceptance driver

Reserve two nodes with four visible GPUs each through your site's scheduler.
Start Ray inside that allocation using the verified controller and consistent
source/environment paths. With Ray 2.56.1, use `--include-dashboard=false` only
with `ray start --head`; omit it from the worker's `ray start --address ...`.
Confirm that both nodes and all eight GPUs are registered.

From the restored NeMo-RL checkout, with the earlier variables available:

```bash
# Reuse an explicitly prepared controller without changing its packages.
export NRL_CONTROLLER_PYTHON=/path/to/verified/controller/bin/python
export RAY_ADDRESS=auto
bash tests/test_suites/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.sh
```

With `NRL_CONTROLLER_PYTHON`, the driver uses offline, no-sync execution through
`uv`. Without it, it uses `uv run --frozen --extra nemo_gym`. The fixed driver
accepts no configuration overrides. It does not allocate nodes, start Ray,
perform the role/library preflight or manage site-specific cleanup. Keep owned
process cleanup and final source checks in your allocation wrapper.

## Acceptance and scope

Inspect `NRL_RUN_DIR/validation.json`, `training.exitcode`, `tee.exitcode`,
`driver.exitcode` and the original logs. Success requires three finite training
steps, positive active-token counts, at least one positive gradient norm, at
least two completed training-time refits, and valid native Gym token/mask/logprob
payloads for every step. Keep `SHA256SUMS`, `metrics.json`, TensorBoard events
and provenance receipts; process exit zero alone is insufficient.

This checks the small training/refit integration, not reward improvement,
convergence, fault recovery, SWE2 results or 128-GPU scale-out. The broader
two-stage results in `docs/guides/swe-rl-qwen3.md` are not SGLang smoke-test results.
