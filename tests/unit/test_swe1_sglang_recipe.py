"""CPU checks for the distinct SWE1 SGLang integration-smoke configuration."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_RECIPE = (
    G_REPO_ROOT / "examples/configs/recipes/llm/"
    "grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-async-gym-sglang-quick.yaml"
)


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch):
    register_omegaconf_resolvers()
    for key, value in {
        "NRL_GYM_VENV_DIR": "/recipe-test/gym-venvs",
        "NRL_MODEL_PATH": "/recipe-test/model",
        "NRL_SWE1_TRAIN_PATH": "/recipe-test/train.jsonl",
        "NRL_SWE1_VALIDATION_PATH": "/recipe-test/validation.jsonl",
        "NRL_RUN_DIR": "/recipe-test/run",
    }.items():
        monkeypatch.setenv(key, value)
    return OmegaConf.to_container(load_config(G_RECIPE), resolve=True)


def test_quick_recipe_allocates_separate_policy_and_generation_nodes(recipe):
    cluster = recipe["cluster"]
    policy = recipe["policy"]
    generation = policy["generation"]
    allocation = generation["colocated"]
    assert not allocation["enabled"]
    assert cluster == {"num_nodes": 2, "gpus_per_node": 4}
    assert allocation["resources"] == {"num_nodes": 1, "gpus_per_node": 4}
    assert generation["sglang_cfg"]["sglang_server_config"]["num_gpus"] == 4
    assert generation["sglang_cfg"]["sglang_server_config"]["num_gpus_per_engine"] == 2
    megatron = policy["megatron_cfg"]
    assert megatron["enabled"]
    assert not policy["dtensor_cfg"]["enabled"]
    assert megatron["tensor_model_parallel_size"] == 2
    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["context_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 4
    assert megatron["expert_tensor_parallel_size"] == 1
    assert megatron["sequence_parallel"]


def test_quick_recipe_uses_current_sglang_knobs_and_bounded_work(recipe):
    generation = recipe["policy"]["generation"]
    sglang = generation["sglang_cfg"]
    assert generation["backend"] == "sglang"
    assert generation["use_async_rollouts"]
    assert generation["refit_transport"] is None
    assert sglang["moe_runner_backend"] == "triton"
    assert sglang["cuda_graph_backend_prefill"] == "disabled"
    assert sglang["disable_cuda_graph"] is False
    assert sglang["quantization"]["scheme"] == "bf16"
    assert (
        sglang["context_length"]
        == recipe["policy"]["max_total_sequence_length"]
        == 14336
    )
    assert generation["max_new_tokens"] == 8192
    assert sglang["context_length"] - generation["max_new_tokens"] == 6144
    assert sglang["allow_auto_truncate"] is False
    assert sglang["sglang_server_config"]["pause_generation_mode"] == "retract"
    assert not sglang["sglang_fault_tolerance_config"]["use_fault_tolerance"]
    assert (
        not {
            "disable_piecewise_cuda_graph",
            "refit_timeout_s",
            "engine_startup_timeout_s",
            "use_fault_tolerance",
        }
        & sglang.keys()
    )
    assert recipe["grpo"]["max_num_steps"] == 3
    assert recipe["grpo"]["async_grpo"]["enabled"]
    assert not recipe["grpo"]["async_grpo"]["in_flight_weight_updates"]
    assert recipe["policy"]["train_global_batch_size"] == 16
    assert recipe["policy"]["refit_buffer_size_gb"] == 1.0


def test_gym_and_policy_share_the_thinking_preserving_template(recipe):
    policy = recipe["policy"]
    gym = recipe["env"]["nemo_gym"]
    model = gym["policy_model"]["responses_api_models"]["sglang_model"]
    assert gym["config_paths"][0] == (
        "responses_api_models/sglang_model/configs/sglang_model_for_training.yaml"
    )
    assert model["sglang_chat_template"] == policy["tokenizer"]["chat_template"]
    assert model["chat_template_kwargs"] == policy["tokenizer"]["chat_template_kwargs"]
    assert model["chat_template_kwargs"]["truncate_history_thinking"] is False
    assert model["sglang_tool_format"] == "hermes"
    assert model["uses_reasoning_parser"]
    legacy_agent = gym["single_step_tool_use_with_argument_comparison_swe"][
        "responses_api_agents"
    ]["tool_simulation_agent"]
    assert legacy_agent["model_server"]["name"] == "policy_model"
    assert legacy_agent["resources_server"]["name"] == (
        "swe_pivot_single_step_tool_use_with_argument_comparison_resources_server"
    )


def test_quick_recipe_uses_explicit_input_and_evidence_paths(recipe):
    assert recipe["policy"]["model_name"] == "/recipe-test/model"
    assert recipe["data"]["train"]["data_path"] == "/recipe-test/train.jsonl"
    assert recipe["data"]["validation"]["data_path"] == "/recipe-test/validation.jsonl"
    assert recipe["logger"]["log_dir"] == "/recipe-test/run/logs"
    assert recipe["env"]["nemo_gym"]["uv_venv_dir"] == "/recipe-test/gym-venvs"
    assert not recipe["logger"]["wandb_enabled"]
    assert recipe["logger"]["tensorboard_enabled"]
    assert not recipe["env"]["should_log_nemo_gym_responses"]
    assert not recipe["policy"]["generation"]["vllm_cfg"]["enable_vllm_metrics_logger"]
    assert not recipe["checkpointing"]["enabled"]
