#!/usr/bin/env bash
set -euo pipefail

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for the Bailian Judge}"

: "${MEMCALIB_MODEL_PATH:?Set MEMCALIB_MODEL_PATH}"
: "${MEMCALIB_TRAIN_FILE:?Set MEMCALIB_TRAIN_FILE}"
: "${MEMCALIB_VAL_FILE:?Set MEMCALIB_VAL_FILE}"
: "${MEMCALIB_EXPERIMENT_NAME:?Set MEMCALIB_EXPERIMENT_NAME}"
: "${MEMCALIB_SAVE_DIR:?Set MEMCALIB_SAVE_DIR}"
: "${MEMCALIB_OUTPUT_DIR:?Set MEMCALIB_OUTPUT_DIR}"

MEMCALIB_METHOD=${MEMCALIB_METHOD:-fine_grained}
MEMCALIB_PROMPT_BATCH_SIZE=${MEMCALIB_PROMPT_BATCH_SIZE:-64}
MEMCALIB_PPO_MINI_BATCH_SIZE=${MEMCALIB_PPO_MINI_BATCH_SIZE:-32}

export TENSORBOARD_DIR="${MEMCALIB_TENSORBOARD_LOG_DIR:-${MEMCALIB_OUTPUT_DIR}/tensorboard}"

python3 -m recipe.memcalib_credit.main \
    actor_rollout_ref.model.path="${MEMCALIB_MODEL_PATH}" \
    data.train_files="${MEMCALIB_TRAIN_FILE}" \
    data.val_files="${MEMCALIB_VAL_FILE}" \
    data.train_batch_size="${MEMCALIB_PROMPT_BATCH_SIZE}" \
    data.max_prompt_length="${MEMCALIB_MAX_PROMPT_LENGTH:-4096}" \
    data.max_response_length="${MEMCALIB_MAX_RESPONSE_LENGTH:-2048}" \
    data.apply_chat_template_kwargs.enable_thinking="${MEMCALIB_ENABLE_THINKING:-False}" \
    actor_rollout_ref.model.use_remove_padding="${MEMCALIB_USE_REMOVE_PADDING:-True}" \
    actor_rollout_ref.actor.optim.lr="${MEMCALIB_LEARNING_RATE:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MEMCALIB_PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MEMCALIB_PPO_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${MEMCALIB_KL_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${MEMCALIB_GPU_MEMORY_UTILIZATION:-0.6}" \
    actor_rollout_ref.rollout.n="${MEMCALIB_GROUP_SIZE:-8}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MEMCALIB_LOGPROB_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MEMCALIB_LOGPROB_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs="${MEMCALIB_EPOCHS:-1}" \
    trainer.project_name="${MEMCALIB_PROJECT_NAME:-memcalib}" \
    trainer.experiment_name="${MEMCALIB_EXPERIMENT_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node="${MEMCALIB_GPUS_PER_NODE:-8}" \
    trainer.nnodes="${MEMCALIB_NNODES:-1}" \
    trainer.save_freq="${MEMCALIB_SAVE_FREQ:-30}" \
    trainer.test_freq="${MEMCALIB_TEST_FREQ:-30}" \
    trainer.val_before_train="${MEMCALIB_VAL_BEFORE_TRAIN:-True}" \
    trainer.default_local_dir="${MEMCALIB_SAVE_DIR}" \
    trainer.rollout_data_dir="${MEMCALIB_OUTPUT_DIR}/rollouts" \
    trainer.validation_data_dir="${MEMCALIB_OUTPUT_DIR}/validation" \
    credit.method="${MEMCALIB_METHOD}" \
    credit.fine_grained.merge_a_overuse="${MEMCALIB_MERGE_A_OVERUSE:-True}" \
    credit.localization.mode="${MEMCALIB_LOCALIZATION_MODE:-positive_only}" \
    credit.localization.granularity="${MEMCALIB_LOCALIZATION_GRANULARITY:-sentence}" \
    credit.trace.enabled="${MEMCALIB_TRACE_ENABLED:-True}" \
    credit.trace.output_dir="${MEMCALIB_OUTPUT_DIR}/credit_trace" \
    credit.judge.judge_model="${MEMCALIB_JUDGE_MODEL:-deepseek-v4-pro}" \
    credit.judge.max_concurrent="${MEMCALIB_JUDGE_CONCURRENCY:-64}" \
    credit.judge.requests_per_minute="${MEMCALIB_JUDGE_RPM:-512}" \
    "$@"
