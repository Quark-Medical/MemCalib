#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# This launcher owns only the Qwen3.5 model, engine, LoRA, and parallel-layout
# overrides. The selected recipe launcher continues to own data, algorithm,
# Judge behavior, logging, checkpoint frequency, and experiment defaults.
: "${MEMCALIB_MODEL_PATH:?Set MEMCALIB_MODEL_PATH}"
export MEMCALIB_LEARNING_RATE=${MEMCALIB_LEARNING_RATE:-5e-6}
export MEMCALIB_NNODES=${MEMCALIB_NNODES:-2}
export MEMCALIB_GPUS_PER_NODE=${MEMCALIB_GPUS_PER_NODE:-8}
# Judge request limits for this launcher.
export MEMCALIB_JUDGE_CONCURRENCY=${MEMCALIB_JUDGE_CONCURRENCY:-144}
export MEMCALIB_JUDGE_RPM=${MEMCALIB_JUDGE_RPM:-512}

# On each 8-GPU student node, TP2 + PP1 and EP8 keep model-parallel groups
# node-local. SGLang TP4 creates two rollout replicas per student node.
MEMCALIB_ACTOR_TP=${MEMCALIB_ACTOR_TP:-2}
MEMCALIB_ACTOR_PP=${MEMCALIB_ACTOR_PP:-1}
MEMCALIB_ACTOR_CP=${MEMCALIB_ACTOR_CP:-1}
MEMCALIB_ACTOR_EP=${MEMCALIB_ACTOR_EP:-8}
MEMCALIB_ACTOR_ETP=${MEMCALIB_ACTOR_ETP:-1}
MEMCALIB_ROLLOUT_TP=${MEMCALIB_ROLLOUT_TP:-4}
MEMCALIB_LORA_RANK=${MEMCALIB_LORA_RANK:-64}
MEMCALIB_LORA_ALPHA=${MEMCALIB_LORA_ALPHA:-64}
MEMCALIB_LORA_TARGET_MODULES=${MEMCALIB_LORA_TARGET_MODULES:-}
MEMCALIB_PPO_MICRO_BATCH_SIZE=${MEMCALIB_PPO_MICRO_BATCH_SIZE:-4}
MEMCALIB_LOGPROB_MICRO_BATCH_SIZE=${MEMCALIB_LOGPROB_MICRO_BATCH_SIZE:-8}
export MEMCALIB_PPO_MINI_BATCH_SIZE=${MEMCALIB_PPO_MINI_BATCH_SIZE:-32}
MEMCALIB_MEGATRON_PARAM_OFFLOAD=${MEMCALIB_MEGATRON_PARAM_OFFLOAD:-False}
MEMCALIB_MEGATRON_OPTIMIZER_OFFLOAD=${MEMCALIB_MEGATRON_OPTIMIZER_OFFLOAD:-False}
MEMCALIB_MEGATRON_GRAD_OFFLOAD=${MEMCALIB_MEGATRON_GRAD_OFFLOAD:-False}
MEMCALIB_MEGATRON_RECOMPUTE=${MEMCALIB_MEGATRON_RECOMPUTE:-False}
export MEMCALIB_GPU_MEMORY_UTILIZATION=${MEMCALIB_GPU_MEMORY_UTILIZATION:-0.5}
MEMCALIB_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${MEMCALIB_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
MEMCALIB_ROLLOUT_MAX_NUM_SEQS=${MEMCALIB_ROLLOUT_MAX_NUM_SEQS:-64}
WEIGHT_UPDATE_BUCKET_MB=${MEMCALIB_WEIGHT_UPDATE_BUCKET_MEGABYTES:-4096}
MEMCALIB_QWEN35_PREFLIGHT_ONLY=${MEMCALIB_QWEN35_PREFLIGHT_ONLY:-False}
MEMCALIB_QWEN35_LAYOUT_ONLY=${MEMCALIB_QWEN35_LAYOUT_ONLY:-False}

MEMCALIB_MAX_PROMPT_LENGTH=${MEMCALIB_MAX_PROMPT_LENGTH:-4096}
MEMCALIB_MAX_RESPONSE_LENGTH=${MEMCALIB_MAX_RESPONSE_LENGTH:-2048}
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
    echo "MEMCALIB_MAX_MODEL_LEN must cover prompt length plus response length" >&2
    exit 2
fi

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export QWEN35_MOE_EXPERT_SCHEMA=${QWEN35_MOE_EXPERT_SCHEMA:-auto}

LAYOUT_ONLY_OVERRIDE=()
case "${MEMCALIB_QWEN35_LAYOUT_ONLY}" in
    1 | true | True | TRUE)
        LAYOUT_ONLY_OVERRIDE+=(--layout-only)
        ;;
    0 | false | False | FALSE)
        ;;
    *)
        echo "MEMCALIB_QWEN35_LAYOUT_ONLY must be true or false" >&2
        exit 2
        ;;
esac

python3 -m recipe.memcalib_credit.qwen35_lora \
    "${LAYOUT_ONLY_OVERRIDE[@]}" \
    --model-path "${MEMCALIB_MODEL_PATH}" \
    --nnodes "${MEMCALIB_NNODES}" \
    --gpus-per-node "${MEMCALIB_GPUS_PER_NODE}" \
    --actor-tp "${MEMCALIB_ACTOR_TP}" \
    --actor-pp "${MEMCALIB_ACTOR_PP}" \
    --actor-cp "${MEMCALIB_ACTOR_CP}" \
    --actor-ep "${MEMCALIB_ACTOR_EP}" \
    --actor-etp "${MEMCALIB_ACTOR_ETP}" \
    --rollout-tp "${MEMCALIB_ROLLOUT_TP}" \
    --ppo-mini-batch-size "${MEMCALIB_PPO_MINI_BATCH_SIZE}" \
    --ppo-micro-batch-size "${MEMCALIB_PPO_MICRO_BATCH_SIZE}"

if ((${#LAYOUT_ONLY_OVERRIDE[@]} > 0)); then
    exit 0
fi

case "${MEMCALIB_QWEN35_PREFLIGHT_ONLY}" in
    1 | true | True | TRUE)
        exit 0
        ;;
    0 | false | False | FALSE)
        ;;
    *)
        echo "MEMCALIB_QWEN35_PREFLIGHT_ONLY must be true or false" >&2
        exit 2
        ;;
esac

LORA_TARGET_OVERRIDE=()
if [[ -n "${MEMCALIB_LORA_TARGET_MODULES}" ]]; then
    LORA_TARGET_OVERRIDE+=(
        "actor_rollout_ref.model.lora.target_modules=${MEMCALIB_LORA_TARGET_MODULES}"
    )
fi

RECOMPUTE_OVERRIDE=()
case "${MEMCALIB_MEGATRON_RECOMPUTE}" in
    1 | true | True | TRUE)
        RECOMPUTE_OVERRIDE+=(
            actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
            actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
            actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        )
        ;;
    0 | false | False | FALSE)
        ;;
    *)
        echo "MEMCALIB_MEGATRON_RECOMPUTE must be true or false" >&2
        exit 2
        ;;
esac

QWEN35_BASE_RUN_SCRIPT=${MEMCALIB_QWEN35_BASE_RUN_SCRIPT:-${SCRIPT_DIR}/run.sh}
if [[ ! -f "${QWEN35_BASE_RUN_SCRIPT}" ]]; then
    echo "Qwen3.5 base run script does not exist: ${QWEN35_BASE_RUN_SCRIPT}" >&2
    exit 2
fi

exec bash "${QWEN35_BASE_RUN_SCRIPT}" \
    trainer.resume_mode=disable \
    model_engine=megatron \
    '~actor_rollout_ref.actor.fsdp_config' \
    '~actor_rollout_ref.ref.fsdp_config' \
    actor_rollout_ref.model.external_lib=recipe.memcalib_credit.compat.qwen35_moe_bridge \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora.rank="${MEMCALIB_LORA_RANK}" \
    actor_rollout_ref.model.lora.alpha="${MEMCALIB_LORA_ALPHA}" \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming \
    "${LORA_TARGET_OVERRIDE[@]}" \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MEMCALIB_PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${MEMCALIB_ACTOR_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${MEMCALIB_ACTOR_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${MEMCALIB_ACTOR_CP}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${MEMCALIB_ACTOR_EP}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${MEMCALIB_ACTOR_ETP}" \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.megatron.param_offload="${MEMCALIB_MEGATRON_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${MEMCALIB_MEGATRON_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.grad_offload="${MEMCALIB_MEGATRON_GRAD_OFFLOAD}" \
    "${RECOMPUTE_OVERRIDE[@]}" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${MEMCALIB_ROLLOUT_TP}" \
    actor_rollout_ref.rollout.max_model_len="${MEMCALIB_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MEMCALIB_ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${MEMCALIB_ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MEMCALIB_LOGPROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_UPDATE_BUCKET_MB}" \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MEMCALIB_LOGPROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${MEMCALIB_ACTOR_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${MEMCALIB_ACTOR_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${MEMCALIB_ACTOR_CP}" \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${MEMCALIB_ACTOR_EP}" \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${MEMCALIB_ACTOR_ETP}" \
    actor_rollout_ref.ref.megatron.param_offload="${MEMCALIB_MEGATRON_PARAM_OFFLOAD}" \
    "$@"
