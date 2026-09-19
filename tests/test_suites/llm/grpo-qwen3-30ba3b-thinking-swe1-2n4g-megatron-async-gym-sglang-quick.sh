#!/bin/bash
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
set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES=120
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONFIG_PATH="$PROJECT_ROOT/examples/configs/recipes/llm/$(basename "$0" .sh).yaml"
# Do not source common.env: its reusable log directory is inappropriate for an
# acceptance run. All evidence below lives in one explicitly fresh directory.
if [[ -n "${TEST_DRYRUN:-}" ]]; then
    test -f "$CONFIG_PATH"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "[ERROR] This fixed three-step acceptance profile does not accept config overrides" >&2
    exit 1
fi
for NAME in NRL_RUN_DIR NRL_MODEL_PATH NRL_MODEL_REVISION NRL_SWE1_TRAIN_PATH NRL_SWE1_VALIDATION_PATH NRL_SWE1_DATA_RECEIPT NRL_GYM_VENV_DIR NRL_MEGATRON_CHECKPOINT_DIR; do
    if [[ -z "${!NAME:-}" ]]; then
        echo "[ERROR] Missing $NAME" >&2
        exit 1
    fi
done
if [[ "$NRL_RUN_DIR" != /* || -e "$NRL_RUN_DIR" || -L "$NRL_RUN_DIR" ]]; then
    echo "[ERROR] NRL_RUN_DIR must be an absolute, not-yet-existing evidence directory" >&2
    exit 1
fi
test -d "$NRL_MEGATRON_CHECKPOINT_DIR"
test -w "$NRL_MEGATRON_CHECKPOINT_DIR"
mkdir "$NRL_RUN_DIR"
mkdir "$NRL_RUN_DIR/logs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}"
# Gym reorders component search roots at import time. Pin components as well as
# its core package when reusing an interpreter with older editable installs.
export NEMO_GYM_EXTRA_ROOTS="$PROJECT_ROOT/3rdparty/Gym-workspace/Gym"
cd "$PROJECT_ROOT"

if [[ -n "${NRL_CONTROLLER_PYTHON:-}" ]]; then
    test -x "$NRL_CONTROLLER_PYTHON"
    # Reuse an explicitly prepared controller without touching its packages.
    # Worker environments remain configured separately by NEMO_RL_VENV_DIR and
    # the recipe's NRL_GYM_VENV_DIR; this is not a SYSTEM-site-packages shortcut.
    RUN=(uv run --offline --no-project --no-sync --python "$NRL_CONTROLLER_PYTHON" python)
else
    RUN=(uv run --frozen --extra nemo_gym python)
fi
CHECK=("${RUN[@]}" tests/swe1_sglang_checks.py)
finish() {
    local status=$?
    trap - EXIT
    set +e
    "${CHECK[@]}" source-after --project "$PROJECT_ROOT" --run-dir "$NRL_RUN_DIR" \
        > "$NRL_RUN_DIR/source-after.log" 2>&1
    local source_status=$?
    if [[ "$source_status" -ne 0 ]]; then
        echo "[ERROR] Final source verification failed; see source-after.log" >&2
        status=1
    fi
    printf '%s\n' "$status" > "$NRL_RUN_DIR/driver.exitcode"
    find "$NRL_RUN_DIR" -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum > "$NRL_RUN_DIR/SHA256SUMS"
    if [[ $? -ne 0 ]]; then status=1; fi
    exit "$status"
}
trap finish EXIT
"${CHECK[@]}" preflight --project "$PROJECT_ROOT" --run-dir "$NRL_RUN_DIR" \
    2>&1 | tee "$NRL_RUN_DIR/preflight.log"

TRAIN=("${RUN[@]}" examples/nemo_gym/run_grpo_nemo_gym.py --config "$CONFIG_PATH")
printf '%q ' "${TRAIN[@]}" > "$NRL_RUN_DIR/command.txt"
printf '\n' >> "$NRL_RUN_DIR/command.txt"
set +e
timeout --signal=TERM --kill-after=60s 110m "${TRAIN[@]}" 2>&1 | tee "$NRL_RUN_DIR/run.log"
TRAIN_STATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${TRAIN_STATUS[0]}" > "$NRL_RUN_DIR/training.exitcode"
printf '%s\n' "${TRAIN_STATUS[1]}" > "$NRL_RUN_DIR/tee.exitcode"
if [[ "${TRAIN_STATUS[0]}" -ne 0 || "${TRAIN_STATUS[1]}" -ne 0 ]]; then
    echo "[ERROR] Training/log capture failed; original evidence retained in $NRL_RUN_DIR" >&2
    exit 1
fi
"${RUN[@]}" tests/json_dump_tb_logs.py "$NRL_RUN_DIR/logs" \
    --output_path "$NRL_RUN_DIR/metrics.json" --error-on-conflicts \
    --require-tag-prefix train/ 2>&1 | tee "$NRL_RUN_DIR/tensorboard-dump.log"
"${CHECK[@]}" validate --project "$PROJECT_ROOT" --run-dir "$NRL_RUN_DIR" \
    2>&1 | tee "$NRL_RUN_DIR/validation.log"
echo "[PASS] Three real optimizer steps and completed refit barriers verified: $NRL_RUN_DIR/validation.json"
