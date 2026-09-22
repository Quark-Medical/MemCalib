#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

: "${MEMCALIB_MODEL_PATH:?Set MEMCALIB_MODEL_PATH}"
export MEMCALIB_USE_REMOVE_PADDING=${MEMCALIB_USE_REMOVE_PADDING:-False}
export MEMCALIB_MAX_PROMPT_LENGTH=${MEMCALIB_MAX_PROMPT_LENGTH:-4096}
export MEMCALIB_MAX_RESPONSE_LENGTH=${MEMCALIB_MAX_RESPONSE_LENGTH:-2048}

if [[ ! "${MEMCALIB_MAX_PROMPT_LENGTH}" =~ ^[0-9]+$ ]] \
    || [[ ! "${MEMCALIB_MAX_RESPONSE_LENGTH}" =~ ^[0-9]+$ ]]; then
    echo "MEMCALIB_MAX_PROMPT_LENGTH and MEMCALIB_MAX_RESPONSE_LENGTH must be non-negative integers" >&2
    exit 2
fi

MEMCALIB_REQUIRED_MODEL_LEN=$((
    10#${MEMCALIB_MAX_PROMPT_LENGTH} + 10#${MEMCALIB_MAX_RESPONSE_LENGTH}
))
MEMCALIB_MAX_MODEL_LEN=${MEMCALIB_MAX_MODEL_LEN:-${MEMCALIB_REQUIRED_MODEL_LEN}}

if [[ ! "${MEMCALIB_MAX_MODEL_LEN}" =~ ^[0-9]+$ ]] \
    || ((10#${MEMCALIB_MAX_MODEL_LEN} < MEMCALIB_REQUIRED_MODEL_LEN)); then
    echo "MEMCALIB_MAX_MODEL_LEN must be an integer no smaller than prompt length plus response length" >&2
    exit 2
fi

exec bash "${SCRIPT_DIR}/run.sh" \
    '~data.apply_chat_template_kwargs.enable_thinking' \
    actor_rollout_ref.model._target_=recipe.memcalib_credit.model_compat.MinistralTextHFModelConfig \
    actor_rollout_ref.model.use_remove_padding="${MEMCALIB_USE_REMOVE_PADDING}" \
    actor_rollout_ref.rollout.max_model_len="${MEMCALIB_MAX_MODEL_LEN}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.config_format=hf \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.text_config.rope_parameters.apply_yarn_scaling=false \
    "$@"
