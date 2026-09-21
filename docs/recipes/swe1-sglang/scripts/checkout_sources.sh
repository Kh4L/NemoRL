#!/usr/bin/env bash
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

# Restore source only. This does not install dependencies or launch training.
set -euo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || fail 'Usage: checkout_sources.sh /absolute/new/NeMo-RL [docs|runtime]'
destination=$1
profile=${2:-docs}
[[ "$profile" == docs || "$profile" == runtime ]] || fail 'Profile must be docs or runtime'
[[ "$destination" == /* && "$destination" != / && "$destination" != */ ]] || fail 'Destination must be absolute, without a trailing slash'
[[ ! -e "$destination" && ! -L "$destination" ]] || fail 'Destination already exists (including symlinks)'
parent=$(dirname -- "$destination")
leaf=$(basename -- "$destination")
[[ "$leaf" != . && "$leaf" != .. && -d "$parent" ]] || fail 'Destination parent must exist and basename must be new'
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f "$recipe/SOURCES.lock" && ! -L "$recipe/SOURCES.lock" ]] || fail 'SOURCES.lock must be a regular file'

# Prevent ambient worktree/index variables from redirecting Git operations.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
export GIT_TERMINAL_PROMPT=0
command -v git >/dev/null || fail 'Git is required'

# Parse allowlisted data without evaluating shell expressions.
seen=' '
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || fail 'Malformed SOURCES.lock line'
    key=${line%%=*}
    value=${line#*=}
    case "$key" in
        FORMAT_VERSION|NEMO_URL|NEMO_RUNTIME|NEMO_DOCS|GYM_URL|GYM_COMMIT) ;;
        *) fail "Unknown SOURCES.lock key: $key" ;;
    esac
    [[ "$seen" != *" $key "* && -n "$value" ]] || fail "Duplicate or empty SOURCES.lock key: $key"
    seen+="$key "
    printf -v "$key" '%s' "$value"
done < "$recipe/SOURCES.lock"
for key in FORMAT_VERSION NEMO_URL NEMO_RUNTIME NEMO_DOCS GYM_URL GYM_COMMIT; do
    [[ "$seen" == *" $key "* ]] || fail "Missing SOURCES.lock key: $key"
done
[[ "$FORMAT_VERSION" == 1 ]] || fail 'Unsupported SOURCES.lock format'
[[ "$NEMO_URL" == https://github.com/Kh4L/NemoRL.git ]] || fail 'Unexpected NeMo public fork URL'
[[ "$GYM_URL" == https://github.com/Kh4L/NemoGym.git ]] || fail 'Unexpected Gym public fork URL'
for key in NEMO_RUNTIME NEMO_DOCS GYM_COMMIT; do
    [[ "${!key}" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid commit: $key"
done
target=$NEMO_DOCS
[[ "$profile" != runtime ]] || target=$NEMO_RUNTIME

umask 077
mkdir -- "$destination"
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "Checkout failed; partial destination retained: %s\n" "$destination" >&2; fi' EXIT
git init "$destination"
git -C "$destination" remote add origin "$NEMO_URL"
git -C "$destination" fetch --no-tags --no-recurse-submodules origin "$target"
[[ "$(git -C "$destination" rev-parse 'FETCH_HEAD^{commit}')" == "$target" ]] || fail 'Fetched NeMo commit differs'
git -C "$destination" checkout --detach "$target"

gym_path=3rdparty/Gym-workspace/Gym
[[ "$(git -C "$destination" rev-parse "HEAD:$gym_path")" == "$GYM_COMMIT" ]] || fail 'Pinned Gym gitlink differs'
# The tested source predates the companion Gym merge. Override only this
# checkout's submodule URL; do not edit .gitmodules or advance its commit.
git -C "$destination" submodule init -- "$gym_path"
git -C "$destination" config "submodule.$gym_path.url" "$GYM_URL"
git -C "$destination" submodule update --init --recursive
[[ "$(git -C "$destination" rev-parse HEAD)" == "$target" ]] || fail 'Final NeMo commit differs'
[[ "$(git -C "$destination/$gym_path" rev-parse HEAD)" == "$GYM_COMMIT" ]] || fail 'Final Gym commit differs'
[[ -z "$(git -C "$destination" status --porcelain --untracked-files=all)" ]] || fail 'NeMo checkout is not clean'
git -C "$destination" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=all)"'
submodules=$(git -C "$destination" submodule status --recursive)
if printf '%s\n' "$submodules" | grep -Eq '^[+U-]'; then
    fail 'A recursive submodule is missing or at the wrong commit'
fi
printf '%s\n' "$submodules"
trap - EXIT
printf 'Source checkout verified: %s\nNeMo: %s\nGym: %s\nNo environments installed or training launched.\n' "$destination" "$target" "$GYM_COMMIT"
