# Validation: bounded SWE1 training and refit

[Recipe and setup](README.md)

On September 19, 2026, the native SWE1/SGLang recipe completed three training
steps on two GB200 nodes with four GPUs each. Megatron training used one node;
SGLang generation used the other. The allocation completed with exit zero in
17 minutes 45 seconds. The environment reused compiled artifacts, including a
retained Transformer Engine wheel; this was not a fresh-install test.

## Measured result

| Step | Native Gym rows | Active training tokens | Gradient norm | Training refit (seconds) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16 | 53,390 | 1.63803792 | 4.66221333 |
| 2 | 16 | 23,883 | 1.25164545 | 4.63286209 |
| 3 | 16 | 20,300 | 0 | 4.71447992 |

All three steps had finite loss and gradients and positive active-token counts.
The saved payloads contain 48 rows in total. Two gradient norms were positive;
step 3's zero gradient is reported unchanged. Three training-time blocking
refit barriers completed; the initial setup refit is not counted.

The committed `tests/swe1_sglang_checks.py` checks were rerun on the saved
metrics, three per-step JSONL payloads and training log. The derived result
matched the driver's validation and the node's independent check. Training,
log capture and driver exit receipts were all zero. Payload checks cover row
counts, token IDs, masks, finite active logprobs, advantages and rewards.
Positive reward or reward improvement was not required for acceptance.

## Runtime checks

- Both nodes passed actual Megatron, async-worker and SGLang import/library
  checks: six observations, each with CUDA available and four visible GPUs.
- Each observation contained seven selected CUDA SONAMEs with matching library
  hashes and dynamic metadata and a single provider per observed SONAME,
  including cuDNN. NVTX was not observed; its runtime loading is not established.
- Actor interpreter identities and environment mappings were checked. Final
  cleanup receipts on both nodes reported no remaining owned processes.
- Source checks passed with two recorded, allowlisted runtime substitutions on
  node 0 and the driver: `dynamic_context.py` and
  `megatron/training/training.py`. Node 1 had no runtime patches. This is not a
  claim of byte-identical source before and after execution.

The run's original artifacts and hashes were retained for verification.
Operational logs and environment artifacts are not distributed with this
guide; this summary is not a portable replay of the original environment.

## Exact source and input pins

| Component | Revision |
| --- | --- |
| Tested NeMo-RL runtime | `e4df35ba3ab3c20906335fa7ff978b9766778179` |
| NeMo-RL documentation follow-up | `9271c3a9718127d02c1c30ede3557c6aee3d95ec` |
| Companion NeMo-Gym | `ea5a37ea8b78a986cdb2e670b14012cd3981df03` |
| SGLang Miles | `3003d70f680d41c59d1b7acbf65cb47795dfd19e` |
| Qwen3-30B-A3B-Thinking-2507 model | `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` |
| Nemotron-RL-Super-Training-Blends dataset | `b90f74f1d0bafeec6d1f1321173f6775ba5bda2e` |

The documentation follow-up changes only `docs/guides/swe-rl-qwen3.md` relative
to the tested runtime. Use the accompanying `SOURCES.lock` and bootstrap;
do not infer equivalent
behavior from a newer fork or upstream commit.

## Limits

This result does not establish fresh dependency compilation, independent engine
tensor or weight-version equality, convergence, reward improvement, fault
recovery, performance efficiency or 128-GPU scale-out. Refit timings establish
completion of the trainer's blocking refit path, not independent inspection of
engine weights.

Both integration subsets come from the prepared SWE1 training split, so they
are not a held-out benchmark. No SWE2/end-to-end agentic result, routed-expert
replay or worker-owned token capture is claimed. The recipe uses the native
Gym generation adapter's token IDs and logprobs.
