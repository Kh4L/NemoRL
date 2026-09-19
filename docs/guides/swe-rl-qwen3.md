# Two-Stage SWE RL for Qwen3-30B-A3B-Thinking

This guide explains how to post-train a **thinking** MoE model — [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) (30B total / 3B active) — into a stronger software-engineering (SWE) agent using a **two-stage RL recipe**.

The recipe follows the SWE portion of the [Nemotron 3 Super recipe](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md) (its Stage 2.1 / 2.2): a single-step **pivot** stage followed by **end-to-end agentic** RL. It is adapted for a Qwen3 *thinking* model in two important ways:

1. **Interleaved thinking via a custom chat template.** A custom serving chat template **preserves reasoning content across agent turns**, so the model performs *interleaved thinking* over a multi-turn trajectory, instead of discarding the chain of thought between turns.
2. **Multi-harness agent environment.** The underlying NeMo-Gym `swe_agents` environment drives the model through **three agent harnesses** — Codex, OpenCode, and OpenHands (CodeAct) — and mixes their prompts during training so the policy generalizes across scaffolds.

The pivot stage uses the **PivotRL** method ([paper](https://arxiv.org/abs/2603.21383)): rather than running full end-to-end rollouts, it applies RL on informative single-turn decision points sampled from expert trajectories, with flexible argument matching as the reward — cheap, fast, and a strong warm-start for the expensive agentic stage.

Both stages run with **Async GRPO** (non-colocated generation) on the Megatron backend through the NeMo-Gym integration. For the foundational concepts, see the [GRPO Guide](grpo.md), the [Async GRPO Guide](async-grpo.md), and the [NeMo-Gym integration design doc](../design-docs/nemo-gym-integration.md).

## The two stages

RL training proceeds in two stages, each starting from the previous stage's checkpoint:

| Stage | Name | Goal | Environment | Data (open source) | Reward | Rollout |
|-------|------|------|-------------|--------------------|--------|---------|
| **1** | **SWE1 (pivot)** | Learn correct tool-call **format and argument selection** | `single_step_tool_use_with_argument_comparison` | `swe1.jsonl` (50,661 samples) | Argument comparison vs. reference action | **Single step**, no sandbox — fast |
| **2** | **SWE2 (e2e agentic)** | Learn to **resolve real GitHub issues** end-to-end | `swe_agents` (OpenHands + sandbox) | `swe2.jsonl` (R2E-Gym subset) | Test-suite pass (resolved) | **Multi-turn** agent in an Apptainer sandbox |

The pivot stage (SWE1) is cheap because it is single-turn and requires no execution sandbox — it teaches the model to emit well-formed tool calls with the right arguments before the expensive end-to-end agentic stage. SWE2 then does full multi-turn agentic RL where the model edits a repository inside a per-instance sandbox and is rewarded when the hidden test suite passes.

:::{note}
Both stages use [`nvidia/Nemotron-RL-Super-Training-Blends`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends): `swe1.jsonl` for the pivot stage and `swe2.jsonl` for the end-to-end stage. Prepare them as in the [Nemotron 3 Super "Download and prepare the data" section](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md#download-and-prepare-the-data).
:::

## What makes this a *thinking* agent

### Interleaved thinking across turns

Qwen3-Thinking emits a `<think>…</think>` block before each response. In a naive multi-turn agent loop the reasoning from earlier turns is discarded, which breaks the chain of thought the model relies on. This recipe keeps it: the policy generation server is configured with a **custom serving chat template** plus a DeepSeek-R1 reasoning parser, and history thinking is **not** truncated:

```yaml
policy:
  tokenizer:
    chat_template_kwargs:
      enable_thinking: true
  generation:
    vllm_cfg:
      enable_thinking: true
      http_server_serving_chat_kwargs:
        enable_auto_tools: true
        tool_parser: hermes
        reasoning_parser: deepseek_r1
        chat_template: |
          ...                       # renders <think>{reasoning_content}</think> for each assistant turn
        default_chat_template_kwargs:
          enable_thinking: true
          truncate_history_thinking: false   # <-- preserve thinking across turns
```

The template re-emits each assistant turn as `<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}` and always opens generation with `<|im_start|>assistant\n<think>\n`. Because `truncate_history_thinking: false`, the reasoning traces from prior turns remain in the context, giving the model **interleaved thinking** over the whole trajectory.

### Three agent harnesses (mixed-prompt training)

The Stage 2 `swe_agents` environment is built on the OpenHands agent framework but exposes **three distinct agent harnesses** through `agent_prompt_overrides`, and trains on a mix of them (`run_with_mixed_prompts: true`):

| Harness | `agent_cls` | Prompt templates |
|---------|-------------|------------------|
| **Codex** | `CodexAgent` | `prompts/codex/*` |
| **OpenCode** | `OpenCodeAgent` | `prompts/opencode/*` |
| **OpenHands (CodeAct)** | `CodeActAgent` | `prompts/openhands/*` |

Training across multiple scaffolds prevents the policy from overfitting to a single tool/prompt convention and improves robustness when the model is later deployed under a different harness. (Validation uses the CodeAct harness only.)

## Async GRPO + MoE setup

Both stages use the same training backbone. Key configuration choices (full configs below):

- **Async GRPO, non-colocated**: generation runs on a separate pool of nodes from training (`policy.generation.colocated.enabled=false`), with `grpo.async_grpo.enabled=true`, `max_trajectory_age_steps=1`, and `in_flight_weight_updates=true`. This is essential for agentic rollouts whose latency varies widely (a multi-turn SWE trajectory can take many minutes).
- **MoE stability**: the router is frozen (`freeze_moe_router=true`) and the auxiliary load-balancing loss is disabled (`moe_aux_loss_coeff=0`), so RL does not perturb expert routing.
- **Megatron parallelism** (per 8-GPU node): `TP×EP×CP×PP` = `2×8×4×2` (SWE1) / `4×8×4×2` (SWE2), with sequence parallel and sequence packing enabled, at `max_total_sequence_length=131072`.
- **No KL penalty** (`reference_policy_kl_penalty=0`), decoupled clipping (`ratio_clip_min=0.2`, `ratio_clip_max=0.28`), constant LR `1e-6`.

## Results

All evaluations are **pass@1 on SWE-bench Verified, 500 test samples**.

| Stage | Step | Resolved (%) | No Patch (%) | Patch Can't Apply (%) |
|-------|------|:---:|:---:|:---:|
| **Origin** (Qwen3-30B-A3B-Thinking) | – | 23.6 | 2.8 | 3.0 |
| **SWE1** (pivot, used as Stage 2 base) | step230 | 30.4 | 1.6 | 2.6 |
| **SWE2** | step40 | 31.0 | 0.8 | 1.8 |
| **SWE2** | step69 | 28.6 | 0.8 | 1.2 |
| **SWE2** | step100 | 30.4 | 1.0 | 1.4 |
| **SWE2** | step118 | **31.2** ★ | 1.2 | 2.2 |
| **SWE2** | step151 | 26.8 | 1.4 | 2.0 |
| **SWE2** | step172 | 29.8 | 0.4 | 0.8 |
| **SWE2** | step190 | 30.8 | 1.0 | 1.2 |

The pivot stage alone lifts the base model from **23.6% → 30.4%** by fixing tool-call formatting (note the drop in "No Patch" and "Patch Can't Apply" rates). End-to-end agentic RL pushes the best checkpoint to **31.2%** (SWE2 step118).

![SWE-bench Verified two-stage progression](../assets/swe_qwen3_swebench_eval.png)

Evaluation uses the [NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) SWE-bench pipeline, adapted from its [`qwen3coder_30b_swebench` test](https://github.com/NVIDIA-NeMo/Skills/blob/main/tests/slurm-tests/qwen3coder_30b_swebench/run_test.py). The HF-converted checkpoint is served on vLLM (`--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1` plus the same thinking-preserving chat template used in training) and run through the `swe_agent` harness with `agent_max_turns=200` and sampling `temperature=0.7, top_p=0.8, top_k=20`.

### Stage 1 (SWE1, pivot) training reward

Single-step tool-use reward climbs steadily from ~0.2 to ~0.55 over the first 250 steps as the model learns valid tool calls and argument selection. The **step 230** checkpoint is taken as the pivot endpoint and used as the Stage 2 base:

![SWE1 training reward](../assets/swe_qwen3_swe1_reward.png)

### Stage 2 (SWE2) training reward & agent turns

End-to-end reward (`train/total_reward/mean`) trends upward while the mean number of agent turns per sample (`train/turns_per_sample/mean`) decreases — the policy resolves issues with fewer, more decisive actions as training progresses:

![SWE2 training reward and turns per sample](../assets/swe_qwen3_swe2_reward_turns.png)

## Launch

Both stages use the NeMo-Gym GRPO entry point:

```bash
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py --config <config.yaml> {overrides}
```

### Prerequisites

- **Container.** Build from the repo [`docker/Dockerfile`](../../docker/Dockerfile), which already installs [Apptainer](https://apptainer.org/) (symlinked as `singularity`) for the Stage 2 sandbox. Then pre-fetch the SWE virtual environments and mount the [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) repo at `3rdparty/Gym-workspace/Gym`, as in [Nemotron 3 Super → "Rebuild the container for SWE"](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md#rebuild-the-container-for-swe).
- **Data**: download and split `swe1.jsonl` / `swe2.jsonl` following the [Nemotron 3 Super "Download and prepare the data" section](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md#download-and-prepare-the-data).
- **Stage 2 sandbox images.** Provide per-instance `.sif` images for the SWE-bench / R2E-Gym environments via `container_formatter`; build and convert them as in [Nemotron 3 Super Stage 2.2](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md#stage-22---swe-2-64-nodes).
- Set `HF_TOKEN`, `WANDB_API_KEY`, and your data/model paths.

### Stage 1 — SWE1 (pivot)

Single-step, no sandbox. The environment is wired through NeMo-Gym `config_paths`:

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/single_step_tool_use_with_argument_comparison/configs/swe_pivot_single_step_tool_use_with_argument_comparison.yaml
```

The recipe values are baked into [`examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml`](../../examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml) — you only need to point it at your data:

```bash
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml \
  data.train.data_path=/path/to/data/swe1/train-split.jsonl \
  data.validation.data_path=/path/to/data/swe1/val-split.jsonl
```

Stage 1 runs with 16 training nodes + 8 generation nodes, `num_prompts_per_step=64`, `num_generations_per_prompt=8`, `train_global_batch_size=512`, TP=2. The best checkpoint (`step230`) is converted to HF format and used as the starting point for Stage 2.

### SWE1 with SGLang: bounded integration starter

The [SGLang SWE1 configuration](../../examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1_sglang.yaml)
uses the native NeMo-Gym SGLang adapter, exact generation token IDs/logprobs,
and asynchronous collection with **barrier weight updates**. It reuses the
thinking-preserving template above. It does not provide routed-expert replay or
worker-owned token capture. The results above are not results from this backend.

Start with the separately named
[two-node, four-GPU-per-node quick recipe](../../examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.yaml),
not the inherited 128-GPU scale-out configuration. The quick profile uses four
Megatron training GPUs (TP2, PP1, CP1, EP4, expert TP1) and four rollout GPUs
(two TP2 engines), 14336 total tokens / 8192 new tokens, and three steps of four
prompts with four generations each. This is an integration check, not a
convergence or full-context benchmark. This completion budget preserves the
6144-token prompt budget: the initial native rollout test exhausted the previous
2048-token completion budget on four of eight unchanged prompts. The larger
completion budget, three-step training and refit acceptance remain unverified.

Prerequisites:

- Use a clean checkout and its exact recursive submodule commits. The Gym
  submodule must include `responses_api_models/sglang_model`; a stock Gym
  checkout without that adapter is not interchangeable.
- Prepare separate controller, SGLang, Megatron, and Gym service environments
  from the checked-out lockfiles. The current async collector and replay buffer
  additionally use the `vllm` extra, even though generation uses SGLang. Do not
  combine conflicting extras in one environment or reuse an old service
  directory merely because it contains `bin/python`.
- This recipe selects **BF16 + `moe_runner_backend: triton`**. The pinned SGLang
  Miles revision is `3003d70f680d41c59d1b7acbf65cb47795dfd19e`; it does not include
  the BF16 FlashInfer TRT-LLM expert-reload fix. Do not silently switch the runner.
- Prepare an immutable local model snapshot and the placeholder-filled SWE1
  pivot training split, not the raw downloaded JSONL.
  The model directory must end in its full revision hash. Run preparation and
  training in compute allocations, not on a shared login host.

Select a small integration-only subset with the same native adapter renderer
used for generation. Choose new output and run directories outside the source
checkout; set the paths below for your system:

```bash
export NRL_MODEL_REVISION=144afc2f379b542fdd4e85a1fcd5e1f79112d95d
export NRL_MODEL_PATH=/path/to/model/snapshots/$NRL_MODEL_REVISION
export NRL_DATASET_REVISION=b90f74f1d0bafeec6d1f1321173f6775ba5bda2e
export NRL_SWE1_SOURCE=/path/to/prepared/swe1/train-split.jsonl
export NRL_PREPARED_DATA=/path/to/new/swe1-smoke-data
export NRL_GYM_VENV_DIR=/path/to/verified/gym-service-venvs
export NRL_RUN_DIR=/path/to/new/swe1-smoke-run
export NRL_MEGATRON_CHECKPOINT_DIR=/path/to/writable/model-conversion-cache
export NEMO_GYM_EXTRA_ROOTS=$PWD/3rdparty/Gym-workspace/Gym

uv run --frozen --extra nemo_gym tools/prepare_swe1_sglang_smoke.py \
  --source "$NRL_SWE1_SOURCE" --output-dir "$NRL_PREPARED_DATA" \
  --model-path "$NRL_MODEL_PATH" --model-revision "$NRL_MODEL_REVISION" \
  --dataset-revision "$NRL_DATASET_REVISION"

export NRL_SWE1_TRAIN_PATH=$NRL_PREPARED_DATA/train.jsonl
export NRL_SWE1_VALIDATION_PATH=$NRL_PREPARED_DATA/validation.jsonl
export NRL_SWE1_DATA_RECEIPT=$NRL_PREPARED_DATA/manifest.json
```

The tool selects 12 training and four validation rows, without modifying their
prompts, labels, or routing. Selection uses prompt length and deduplication, not
reward. It fails if too few complete prompts fit the 6144-token input budget;
do not truncate examples to force a pass. Use the prepared training split: only
three of the 100 cached validation prompts fit this budget with the pinned model
and renderer. Both integration subsets come from the training split; their names
match the configuration and do not make them a held-out evaluation. Do not use
them to report generalization or benchmark quality. The manifest records the
exact renderer, template, source/split hashes, source line numbers, and rendered
token counts.

With the two-node Ray cluster already running inside your allocation and all
worker/service environments prepared, run the fixed acceptance driver:

```bash
export RAY_ADDRESS=auto
bash tests/test_suites/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.sh
```

For an explicitly prepared, read-only controller, set `NRL_CONTROLLER_PYTHON`
to its interpreter; the driver then uses offline, no-sync execution. This does
not replace the separate worker environments configured by `NEMO_RL_VENV_DIR`
and `NRL_GYM_VENV_DIR`.

Pinning `NEMO_GYM_EXTRA_ROOTS` is important when reusing an interpreter with older
editable installs: checking `nemo_gym.__file__` alone does not establish which
model/resource components Python loads. The preparation and acceptance checks
also verify the actual adapter sources.

Success requires three completed steps with finite loss/gradients, active
training tokens and at least one positive gradient norm; at least two completed
refit barriers; and valid per-step Gym token, loss-mask and logprob artifacts.
The evidence directory preserves the command, source/model/data provenance,
original logs and TensorBoard events, conflict-checked metrics, exit codes,
`validation.json`, and file hashes. Refit timings prove completion of the
trainer's blocking refit path, not independently queried engine weight versions.
An exit-zero process, dry run, or CPU test result alone is not GPU validation.

### Stage 2 — SWE2 (end-to-end agentic)

Multi-turn OpenHands agent in a sandbox. The environment swaps in the `swe_agents` config and sets the agent budget:

```yaml
env:
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml
    swe_agents_train:
      responses_api_agents:
        swe_agents:
          run_with_mixed_prompts: true       # Codex + OpenCode + CodeAct
          container_formatter: [ "/path/to/images/...{instance_id}.sif", ... ]
```

Use [`examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml`](../../examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml), pointing `policy.model_name` at the Stage 1 checkpoint, the data at the `swe2` split, and `container_formatter` at your `.sif` images:

```bash
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml \
  policy.model_name=/path/to/swe1_checkpoint_hf \
  data.train.data_path=/path/to/data/swe2/train-split.jsonl \
  data.validation.data_path=/path/to/data/swe2/val-split.jsonl
```

Stage 2 uses a smaller global batch (`num_prompts_per_step=8`, `train_global_batch_size=64`) because each rollout is an expensive multi-turn trajectory (`agent_max_turns=200`, `swebench_agent_timeout=1800s`), and bumps Megatron TP to 4. `policy.model_name` points at the HF-converted SWE1 checkpoint.

## [Alternative] Using TensorRT-LLM generation backend in Stage 2

The recipe also supports running Stage 2 generation with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) instead of vLLM. This can be useful when TensorRT-LLM's in-flight weight update path offers better throughput for a given cluster configuration.

Use [`examples/swe_bench/grpo_qwen3_30b_async_swe_trtllm.yaml`](../../examples/swe_bench/grpo_qwen3_30b_async_swe_trtllm.yaml), which inherits the SWE2 training and environment settings and switches the generation backend. Set `container_formatter` in the YAML to your `.sif` image template (same as Stage 2 vLLM), then launch with your Stage 1 checkpoint and data:

```bash
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/swe_bench/grpo_qwen3_30b_async_swe_trtllm.yaml \
  policy.model_name=/path/to/swe1_checkpoint_hf \
  data.train.data_path=/path/to/data/swe2/train-split.jsonl \
  data.validation.data_path=/path/to/data/swe2/val-split.jsonl
```

## What to monitor

- **`train/total_reward/mean`** — primary signal and the checkpointing metric for both stages.
- **`train/turns_per_sample/mean`** (Stage 2) — a falling trend indicates the agent is solving tasks more efficiently; a runaway-high value suggests the agent is thrashing or hitting the turn limit.
- **No-Patch / Patch-Can't-Apply rates** during eval — these should fall after the pivot stage if tool-call formatting has been learned.

## References

- [PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost](https://arxiv.org/abs/2603.21383) — the pivot-stage method (NVIDIA).
- [Nemotron 3 Super recipe](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md) — the multi-stage RLVR + SWE + RLHF recipe this is adapted from (data prep, SWE container, sandbox `.sif` build).
- [NeMo-Skills SWE-bench eval](https://github.com/NVIDIA-NeMo/Skills/blob/main/tests/slurm-tests/qwen3coder_30b_swebench/run_test.py) — the evaluation pipeline adapted for these results.
- [Async GRPO Guide](async-grpo.md) and [NeMo-Gym integration](../design-docs/nemo-gym-integration.md).
- [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) — agent/resource servers (`swe_agents`, `single_step_tool_use_with_argument_comparison`).
- [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) · [Nemotron-RL-Super-Training-Blends dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) (`swe1.jsonl`, `swe2.jsonl`).
