"""VeRL trainer hook for MemCalib fine-grained token advantages."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data
from verl.workers.utils.padding import (
    left_right_2_no_padding,
    no_padding_2_padding,
    response_to_nested,
)

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from recipe.memcalib_credit.credit import (
    DEFAULT_ETAS,
    POSITIVE_ONLY_LOCALIZATION,
    LOCALIZATION_GRANULARITIES,
    SENTENCE_LOCALIZATION,
    AdaptiveThresholds,
    channel_localization_signs,
    channel_localization_transforms,
    channel_layout,
    channel_rewards,
    effective_channel_weights,
    group_channel_advantages,
    group_scalar_advantages,
    localize_advantage,
)
from recipe.memcalib_credit.data import build_counterfactual, build_messages
from recipe.memcalib_credit.metrics import (
    FILTERED_VALIDATION_FIELDS,
    add_group_metrics,
    add_judge_metrics,
    add_usage_metrics,
    scalar_group_fraction,
    summarize_valid_only_source_metrics,
    summarize_validation_calibration,
)
from recipe.memcalib_credit.model_compat import (
    load_ministral_text_tokenizer,
    uses_ministral_text_config,
)
from recipe.memcalib_credit.segments import segment_token_ids
from recipe.memcalib_credit.trace import (
    CreditTraceWriter,
    build_credit_trace_records,
    summarize_localization,
)


logger = logging.getLogger(__name__)


def _judge_error_counts(errors) -> dict[str, int]:
    counts = defaultdict(int)
    for error in errors:
        counts[str(error).split(":", 1)[0] or "unknown"] += 1
    return dict(counts)


class MemCalibSyncTrainer(PPOTrainer):
    def _reference_is_colocated_with_actor(self) -> bool:
        model_config = self.config.actor_rollout_ref.model
        lora_rank = model_config.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = model_config.get("lora_rank", 0)
        return lora_rank > 0 or model_config.get("lora_adapter_path") is not None

    def _call_without_separate_reference_worker(self, callback):
        """Skip upstream's separate-ref access when LoRA supplies the base ref."""

        if not (
            self.use_reference_policy and self._reference_is_colocated_with_actor()
        ):
            return callback()

        self.use_reference_policy = False
        try:
            return callback()
        finally:
            self.use_reference_policy = True

    def init_workers(self):
        # main_ppo_sync computes ref_in_actor for LoRA, but verl 0.8 still
        # indexes the absent actor_rollout_ref worker unconditionally.  Keep
        # reference log-prob enabled for training while suppressing only that
        # separate-worker initialization path.
        return self._call_without_separate_reference_worker(
            super().init_workers
        )

    def _start_profiling(self) -> None:
        return self._call_without_separate_reference_worker(
            super()._start_profiling
        )

    def _stop_profiling(self) -> None:
        return self._call_without_separate_reference_worker(
            super()._stop_profiling
        )

    def _init_tokenizer(self):
        model_config = self.config.actor_rollout_ref.model
        if not uses_ministral_text_config(model_config):
            return super()._init_tokenizer()

        # Keep Qwen and all other models on verl's unmodified initialization
        # path. The Ministral launcher opts into this text-only adapter.
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            model_config.path,
            use_shm=model_config.get("use_shm", False),
        )
        self.tokenizer = load_ministral_text_tokenizer(
            local_path,
            trust_remote_code=self.config.data.get("trust_remote_code", False),
        )
        self.processor = None

    def _compute_metrics(
        self, batch, metrics, timing_raw, global_steps, epoch
    ) -> None:
        super()._compute_metrics(
            batch, metrics, timing_raw, global_steps, epoch
        )

        lengths = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses"],
        )
        prompt_lengths = np.asarray(
            lengths["prompts"].offsets().diff().tolist(), dtype=np.int64
        )
        response_lengths = np.asarray(
            lengths["responses"].offsets().diff().tolist(), dtype=np.int64
        )
        non_padding_mask = np.asarray(
            [not tag.get("is_padding", False) for tag in batch.tags], dtype=bool
        )
        if non_padding_mask.any():
            prompt_lengths = prompt_lengths[non_padding_mask]
            response_lengths = response_lengths[non_padding_mask]

        metrics["prompt_length/clip_ratio"] = float(
            np.mean(prompt_lengths == int(self.config.data.max_prompt_length))
        )
        metrics["response_length/clip_ratio"] = float(
            np.mean(response_lengths == int(self.config.data.max_response_length))
        )

    def __init__(self, *args, credit_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit = credit_config or {}
        self.method = str(self.credit.get("method", "fine_grained"))
        if self.method not in {"fine_grained", "gdpo", "grpo"}:
            raise ValueError(f"Unknown credit.method={self.method}")
        fine_grained_config = self.credit.get("fine_grained", {})
        self.layout = channel_layout(
            self.method,
            fine_grained_merge_a_overuse=bool(
                fine_grained_config.get("merge_a_overuse", True)
            ),
        )
        self.channels = self.layout.channels
        localization_config = self.credit.get("localization", {})
        self.localization_mode = str(
            localization_config.get("mode", POSITIVE_ONLY_LOCALIZATION)
        )
        self.localization_granularity = str(
            localization_config.get("granularity", SENTENCE_LOCALIZATION)
        )
        if self.localization_granularity not in LOCALIZATION_GRANULARITIES:
            choices = ", ".join(sorted(LOCALIZATION_GRANULARITIES))
            raise ValueError(
                "Unknown credit.localization.granularity="
                f"{self.localization_granularity}; expected one of {choices}"
            )
        self.localization_transforms = channel_localization_transforms(
            self.layout,
            mode=self.localization_mode,
        )
        self.localization_signs = channel_localization_signs(
            self.layout,
            mode=self.localization_mode,
        )
        self.local_channels = frozenset(self.localization_transforms)

        threshold = self.credit.get("threshold", {})
        self.thresholds = AdaptiveThresholds(
            initial_token=float(
                threshold.get("initial_token", threshold.get("absolute_token", 0.02))
            ),
            initial_sentence=float(
                threshold.get(
                    "initial_sentence", threshold.get("absolute_sentence", 0.01)
                )
            ),
            channels=self.local_channels,
        )
        self._credit_metrics: dict[str, float] = {}
        self._chat_template_kwargs = OmegaConf.to_container(
            self.config.data.get("apply_chat_template_kwargs", {}),
            resolve=True,
        )
        self._credit_trace_writer: CreditTraceWriter | None = None
        trace_config = self.credit.get("trace", {})
        if bool(trace_config.get("enabled", False)):
            configured_dir = trace_config.get("output_dir")
            output_dir = str(configured_dir).strip() if configured_dir else ""
            if not output_dir:
                rollout_dir = str(
                    self.config.trainer.get("rollout_data_dir", "")
                ).strip()
                if rollout_dir:
                    output_dir = os.path.join(
                        os.path.dirname(rollout_dir), "credit_trace"
                    )
            if not output_dir:
                logger.warning(
                    "credit.trace.enabled=true but no output directory is set; "
                    "credit trace is disabled"
                )
            else:
                try:
                    self._credit_trace_writer = CreditTraceWriter(
                        output_dir,
                        compression_level=int(
                            trace_config.get("compression_level", 1)
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "Credit trace initialization failed; tracing is disabled: "
                        "%s: %s",
                        type(exc).__name__,
                        exc,
                    )

    @property
    def _weights(self) -> torch.Tensor:
        weights = effective_channel_weights(
            self.credit.get("channel_weights", {}),
            channels=self.channels,
        )
        return torch.tensor(
            [weights[channel] for channel in self.channels],
            dtype=torch.float32,
        )

    def _records(self, batch: DataProto) -> list[dict[str, Any]]:
        records = []
        for reward_model in batch.non_tensor_batch["reward_model"]:
            if isinstance(reward_model, np.ndarray):
                reward_model = reward_model.item()
            ground_truth = reward_model["ground_truth"]
            records.append(
                json.loads(ground_truth)
                if isinstance(ground_truth, str)
                else ground_truth
            )
        return records

    def _prepare_rewards(
        self, batch: DataProto
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, list[str]]],
        torch.Tensor,
        np.ndarray,
    ]:
        records = self._records(batch)
        valid = np.asarray(
            batch.non_tensor_batch["memcalib_reward_valid"], dtype=bool
        ).copy()
        judgment_json = batch.non_tensor_batch["memcalib_judgments"]
        sets = [{channel: [] for channel in self.channels} for _ in records]
        rewards = torch.zeros(
            (len(records), len(self.channels)),
            dtype=torch.float32,
        )
        kappa_a = float(self.credit.get("kappa_a", 2.0))

        for index, record in enumerate(records):
            if not valid[index]:
                continue
            try:
                raw = judgment_json[index]
                judgments = json.loads(str(raw))
                row_sets, row_rewards = channel_rewards(
                    record,
                    judgments,
                    kappa_a=kappa_a,
                    layout=self.layout,
                )
                sets[index] = row_sets
                rewards[index] = torch.tensor(
                    [row_rewards[channel] for channel in self.channels]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                valid[index] = False
                logger.warning(
                    "Invalid reward inputs for %s: %s: %s",
                    record.get("id"),
                    type(exc).__name__,
                    exc,
                )
        return records, sets, rewards, valid

    def _encode_counterfactual(
        self,
        batch: DataProto,
        row: int,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self._chat_template_kwargs,
        )
        encoded = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = postprocess_data(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=int(self.config.data.max_prompt_length),
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        responses = batch.batch["responses"][row].unsqueeze(0)
        response_mask = batch.batch["response_mask"][row].unsqueeze(0)
        input_ids = torch.cat((prompt_ids.cpu(), responses.cpu()), dim=-1)
        attention_mask = torch.cat(
            (prompt_mask.cpu(), response_mask.cpu()), dim=-1
        )
        return {
            "prompts": prompt_ids.cpu(),
            "responses": responses.cpu(),
            "response_mask": response_mask.cpu(),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }

    def _score_counterfactuals(
        self,
        batch: DataProto,
        records: list[dict[str, Any]],
        channel_sets: list[dict[str, list[str]]],
        valid: np.ndarray,
    ) -> tuple[
        dict[tuple[int, str], torch.Tensor],
        dict[tuple[int, str], str],
        dict[str, float],
    ]:
        construction_started = time.monotonic()
        pending: list[tuple[int, str, dict[str, torch.Tensor]]] = []
        failures: dict[tuple[int, str], str] = {}
        requested = 0
        for row, record in enumerate(records):
            if not valid[row]:
                continue
            for channel in self.channels:
                if (
                    channel not in self.local_channels
                    or not channel_sets[row][channel]
                ):
                    continue
                requested += 1
                key = (row, channel)
                try:
                    overrides, omitted = build_counterfactual(
                        record, channel_sets[row][channel]
                    )
                    messages = build_messages(
                        record,
                        block_overrides=overrides,
                        omitted_parent_ids=omitted,
                    )
                    pending.append(
                        (row, channel, self._encode_counterfactual(batch, row, messages))
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    failures[key] = f"construction:{type(exc).__name__}"
        construction_s = time.monotonic() - construction_started

        scored: dict[tuple[int, str], torch.Tensor] = {}
        chunk_size = int(self.credit.get("counterfactual_batch_size", 256))
        scoring_divisor = self.actor_rollout_wg.world_size * int(
            self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
        )
        scoring_started = time.monotonic()
        chunk_count = 0
        failed_chunks = 0
        for offset in range(0, len(pending), chunk_size):
            chunk_count += 1
            chunk = pending[offset : offset + chunk_size]
            tensors = {
                key: torch.cat([item[2][key] for item in chunk], dim=0)
                for key in (
                    "prompts",
                    "responses",
                    "response_mask",
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                )
            }
            data = DataProto.from_dict(
                tensors=tensors,
            )
            padded, pad_size = pad_dataproto_to_divisor(
                data, scoring_divisor
            )
            try:
                batch_td = left_right_2_no_padding(padded.to_tensordict())
                tu.assign_non_tensor(
                    batch_td,
                    calculate_entropy=False,
                    compute_loss=False,
                    temperature=self.config.actor_rollout_ref.rollout.temperature,
                )
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
                log_probs = no_padding_2_padding(
                    tu.get(output, "log_probs"), batch_td
                ).float()
                output = DataProto.from_dict(
                    tensors={"old_log_probs": log_probs}
                )
                output = unpad_dataproto(output, pad_size)
                log_probs = output.batch["old_log_probs"]
                for index, (row, channel, _) in enumerate(chunk):
                    scored[(row, channel)] = log_probs[index].detach().cpu()
            except Exception as exc:
                failed_chunks += 1
                logger.warning(
                    "Counterfactual scoring chunk failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                for row, channel, _ in chunk:
                    failures[(row, channel)] = f"scoring:{type(exc).__name__}"
        scoring_s = time.monotonic() - scoring_started
        construction_failures = sum(
            reason.startswith("construction:") for reason in failures.values()
        )
        scoring_failures = sum(
            reason.startswith("scoring:") for reason in failures.values()
        )
        return scored, failures, {
            "credit/counterfactual_requested_count": float(requested),
            "credit/counterfactual_constructed_count": float(len(pending)),
            "credit/counterfactual_scored_count": float(len(scored)),
            "credit/counterfactual_construction_failure_count": float(
                construction_failures
            ),
            "credit/counterfactual_scoring_failure_count": float(scoring_failures),
            "credit/counterfactual_success_fraction": (
                len(scored) / requested if requested else 0.0
            ),
            "credit/counterfactual_chunk_count": float(chunk_count),
            "credit/counterfactual_failed_chunk_count": float(failed_chunks),
            "credit/counterfactual_construction_s": construction_s,
            "credit/counterfactual_scoring_s": scoring_s,
        }

    def _segments(
        self, batch: DataProto, row: int, valid_length: int
    ) -> tuple[list[list[int]], bool]:
        token_ids = (
            batch.batch["responses"][row, :valid_length].detach().cpu().tolist()
        )
        segment_cfg = self.credit.get("segment", {})
        try:
            return (
                segment_token_ids(
                    self.tokenizer,
                    token_ids,
                    min_tokens=int(segment_cfg.get("min_tokens", 4)),
                    max_tokens=int(segment_cfg.get("max_tokens", 96)),
                ),
                False,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Response segmentation failed; using one unit: %s", exc)
            return [list(range(valid_length))], True

    def _compute_credit_advantage(self, batch: DataProto) -> DataProto:
        records, channel_sets, rewards, valid = self._prepare_rewards(batch)
        if len(valid) and not valid.any():
            errors = _judge_error_counts(
                batch.non_tensor_batch.get(
                    "memcalib_judge_error", np.full(len(valid), "unknown")
                )
            )
            raise RuntimeError(
                "All Judge rewards in this rollout batch are invalid; "
                f"refusing an empty actor update. errors={errors}"
            )
        response_mask = batch.batch["response_mask"].float()
        device = response_mask.device
        rewards = rewards.to(device)
        weights = self._weights.to(device)
        uids = batch.non_tensor_batch["uid"]
        epsilon = float(self.credit.get("epsilon_group", 1e-6))
        sigma_min = float(self.credit.get("sigma_min", 1e-6))
        trace_enabled = self._credit_trace_writer is not None
        trace_response_lengths = (
            [
                int(value)
                for value in response_mask.sum(dim=1).detach().cpu().tolist()
            ]
            if trace_enabled
            else []
        )
        thresholds_used = self.thresholds.state_dict() if trace_enabled else {}
        channel_advantages: torch.Tensor | None = None
        scalar_rewards: torch.Tensor | None = None
        scalar_advantages: torch.Tensor | None = None
        sequence_advantages: torch.Tensor | None = None
        rms_scale = 1.0
        segment_cache: dict[int, list[list[int]]] = {}
        segment_fallback_rows: dict[int, bool] = {}
        localization_results: dict[tuple[int, str], dict[str, Any]] = {}

        metrics: dict[str, float] = {
            "credit/valid_fraction": float(valid.mean()) if len(valid) else 0.0,
            "credit/invalid_count": float((~valid).sum()),
        }
        add_judge_metrics(metrics, batch, valid)
        add_usage_metrics(metrics, batch, records, valid)
        add_group_metrics(
            metrics,
            rewards,
            uids,
            valid,
            sigma_min,
            self.channels,
        )
        valid_count = max(1, int(valid.sum()))
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
        for channel_index, channel in enumerate(self.channels):
            values = rewards[:, channel_index]
            metrics[f"credit/reward_mean_{channel}"] = (
                float(values[valid_t].mean()) if valid.any() else 0.0
            )
            metrics[f"credit/reward_std_{channel}"] = (
                float(values[valid_t].std(unbiased=False))
                if valid.any()
                else 0.0
            )
            metrics[f"credit/trigger_rate_{channel}"] = (
                sum(bool(row[channel]) for row, keep in zip(channel_sets, valid) if keep)
                / valid_count
            )
            if (
                self.localization_mode != POSITIVE_ONLY_LOCALIZATION
                and channel in self.localization_signs
            ):
                metrics[f"credit/localization_sign_{channel}"] = float(
                    self.localization_signs[channel]
                )

        if self.method == "grpo":
            scalar_rewards = rewards @ weights
            metrics["credit/nondegenerate_group_fraction_scalar"] = (
                scalar_group_fraction(
                    scalar_rewards, uids, valid, sigma_min
                )
            )
            scalar_advantages = group_scalar_advantages(
                scalar_rewards,
                uids,
                valid,
                epsilon=epsilon,
                sigma_min=sigma_min,
            )
            sequence_advantages = scalar_advantages
            advantages = scalar_advantages.unsqueeze(-1) * response_mask
            metrics["credit/rms_scale"] = 1.0
            metrics["credit/localized_fraction"] = 0.0
            metrics["credit/fallback_fraction"] = 0.0
        else:
            channel_advantages = group_channel_advantages(
                rewards,
                uids,
                valid,
                epsilon=epsilon,
                sigma_min=sigma_min,
            )
            sequence_advantages = channel_advantages @ weights
            epsilon_scale = float(self.credit.get("epsilon_scale", 1e-6))
            rms = (
                torch.sqrt(sequence_advantages[valid_t].square().mean() + epsilon_scale)
                if valid.any()
                else torch.tensor(1.0, device=device)
            )
            rms_scale = float(rms)
            advantages = (
                sequence_advantages.unsqueeze(-1) / rms
            ) * response_mask
            metrics["credit/rms_scale"] = float(rms)
            metrics["credit/sequence_advantage_mean"] = (
                float(sequence_advantages[valid_t].mean()) if valid.any() else 0.0
            )
            for channel_index, channel in enumerate(self.channels):
                values = channel_advantages[:, channel_index]
                metrics[f"credit/advantage_rms_{channel}"] = (
                    float(torch.sqrt(values[valid_t].square().mean()))
                    if valid.any()
                    else 0.0
                )
            sequence_indices = [
                index
                for index, channel in enumerate(self.channels)
                if channel not in self.local_channels
            ]
            sequence_only = (
                channel_advantages[:, sequence_indices]
                @ weights[sequence_indices]
            )
            metrics["credit/sequence_only_abs_contribution"] = (
                float(sequence_only[valid_t].abs().mean())
                if valid.any()
                else 0.0
            )

            localized = 0
            fallback = 0
            localized_by_channel = defaultdict(int)
            fallback_by_channel = defaultdict(int)
            direction_gate_by_channel = defaultdict(int)
            absolute_gate_by_channel = defaultdict(int)
            q_positive_by_channel = defaultdict(int)
            fallback_reasons = defaultdict(int)
            max_multiplier = 1.0
            max_conservation_error = 0.0
            multiplier_cap_hits = 0
            direction_gate_open = 0
            absolute_gate_open = 0
            q_positive = 0
            q_total_sum = 0.0
            segmented_responses = 0
            segment_fallbacks = 0
            token_pools: dict[str, list[float]] = defaultdict(list)
            sentence_pools: dict[str, list[float]] = defaultdict(list)
            if self.method == "fine_grained":
                scored, ablation_failures, counterfactual_metrics = (
                    self._score_counterfactuals(
                        batch, records, channel_sets, valid
                    )
                )
                metrics.update(counterfactual_metrics)
                localization_started = time.monotonic()
                localization_cfg = self.credit.get("localization", {})
                threshold_cfg = self.credit.get("threshold", {})
                multiplier_cap = float(
                    localization_cfg.get("multiplier_cap", 4.0)
                )
                configured_etas = localization_cfg.get("eta", {})
                etas = {
                    channel: float(configured_etas.get(channel, DEFAULT_ETAS[channel]))
                    for channel in self.local_channels
                }
                for row in range(len(records)):
                    if not valid[row]:
                        continue
                    valid_length = int(response_mask[row].sum().item())
                    if valid_length <= 0:
                        valid[row] = False
                        continue
                    if (
                        self.localization_granularity == SENTENCE_LOCALIZATION
                        or trace_enabled
                    ):
                        segments, used_fallback = self._segments(
                            batch, row, valid_length
                        )
                        segmented_responses += 1
                        segment_fallbacks += int(used_fallback)
                    else:
                        segments = [list(range(valid_length))]
                        used_fallback = False
                    segment_cache[row] = segments
                    if trace_enabled:
                        segment_fallback_rows[row] = used_fallback
                    for channel_index, channel in enumerate(self.channels):
                        if (
                            channel not in self.local_channels
                            or not channel_sets[row][channel]
                        ):
                            continue
                        key = (row, channel)
                        result = localize_advantage(
                            channel=channel,
                            reward=float(rewards[row, channel_index]),
                            advantage=float(channel_advantages[row, channel_index]),
                            full_log_probs=batch.batch["old_log_probs"][
                                row, :valid_length
                            ].detach().cpu(),
                            ablated_log_probs=(
                                scored[key][:valid_length] if key in scored else None
                            ),
                            segments=segment_cache[row],
                            token_threshold=self.thresholds.values[channel]["token"],
                            sentence_threshold=self.thresholds.values[channel][
                                "sentence"
                            ],
                            absolute_token_threshold=float(
                                threshold_cfg.get("absolute_token", 0.02)
                            ),
                            absolute_sentence_threshold=float(
                                threshold_cfg.get("absolute_sentence", 0.01)
                            ),
                            delta_clip=float(
                                localization_cfg.get("delta_clip", 3.0)
                            ),
                            coverage_gamma=float(
                                localization_cfg.get("coverage_gamma", 1.0)
                            ),
                            eta=etas[channel],
                            multiplier_cap=multiplier_cap,
                            fallback_reason=ablation_failures.get(key, ""),
                            localization_transform=(
                                self.localization_transforms[channel]
                            ),
                            granularity=self.localization_granularity,
                        )
                        if trace_enabled:
                            localization_results[key] = summarize_localization(
                                result, segment_cache[row]
                            )
                        uniform = channel_advantages[row, channel_index]
                        correction = (
                            result.token_advantages.to(device) - uniform
                        ) * weights[channel_index] / rms
                        advantages[row, :valid_length] += correction
                        token_pools[channel].extend(result.token_pool)
                        if (
                            self.localization_granularity
                            == SENTENCE_LOCALIZATION
                        ):
                            sentence_pools[channel].extend(
                                result.sentence_pool
                            )
                        localized += int(result.localized)
                        fallback += int(not result.localized)
                        localized_by_channel[channel] += int(result.localized)
                        fallback_by_channel[channel] += int(not result.localized)
                        direction_gate_open += int(result.direction_gate)
                        absolute_gate_open += int(result.absolute_gate)
                        q_positive += int(result.q_total > 0)
                        q_total_sum += result.q_total
                        direction_gate_by_channel[channel] += int(
                            result.direction_gate
                        )
                        absolute_gate_by_channel[channel] += int(
                            result.absolute_gate
                        )
                        q_positive_by_channel[channel] += int(result.q_total > 0)
                        if not result.localized:
                            reason = result.fallback.split(":", 1)[0]
                            fallback_reasons[reason] += 1
                        multiplier_cap_hits += int(
                            result.localized
                            and result.max_multiplier >= multiplier_cap - 1e-6
                        )
                        max_multiplier = max(
                            max_multiplier, result.max_multiplier
                        )
                        max_conservation_error = max(
                            max_conservation_error, result.conservation_error
                        )

                min_token_samples = int(
                    threshold_cfg.get("min_token_samples", 256)
                )
                min_sentence_samples = int(
                    threshold_cfg.get("min_sentence_samples", 32)
                )
                for channel in self.local_channels:
                    metrics[f"credit/token_pool_size_{channel}"] = float(
                        len(token_pools[channel])
                    )
                    metrics[f"credit/token_threshold_update_ready_{channel}"] = float(
                        len(token_pools[channel]) >= min_token_samples
                    )
                    if (
                        self.localization_granularity
                        == SENTENCE_LOCALIZATION
                    ):
                        metrics[f"credit/sentence_pool_size_{channel}"] = float(
                            len(sentence_pools[channel])
                        )
                        metrics[
                            f"credit/sentence_threshold_update_ready_{channel}"
                        ] = float(
                            len(sentence_pools[channel]) >= min_sentence_samples
                        )

                self.thresholds.update(
                    token_pools,
                    sentence_pools,
                    token_quantile=float(
                        threshold_cfg.get("token_quantile", 0.75)
                    ),
                    sentence_quantile=float(
                        threshold_cfg.get("sentence_quantile", 0.70)
                    ),
                    absolute_token=float(
                        threshold_cfg.get("absolute_token", 0.02)
                    ),
                    absolute_sentence=float(
                        threshold_cfg.get("absolute_sentence", 0.01)
                    ),
                    ema=float(threshold_cfg.get("ema", 0.9)),
                    min_token_samples=min_token_samples,
                    min_sentence_samples=min_sentence_samples,
                )
                metrics["credit/localization_s"] = (
                    time.monotonic() - localization_started
                )

            localization_total = localized + fallback
            metrics["credit/localized_fraction"] = (
                localized / localization_total if localization_total else 0.0
            )
            metrics["credit/fallback_fraction"] = (
                fallback / localization_total if localization_total else 0.0
            )
            metrics["credit/localization_candidate_count"] = float(
                localization_total
            )
            metrics["credit/direction_gate_open_fraction"] = (
                direction_gate_open / localization_total
                if localization_total
                else 0.0
            )
            metrics["credit/direction_gate_closed_fraction"] = (
                (localization_total - direction_gate_open) / localization_total
                if localization_total
                else 0.0
            )
            metrics["credit/absolute_gate_open_fraction"] = (
                absolute_gate_open / localization_total
                if localization_total
                else 0.0
            )
            metrics["credit/positive_q_fraction"] = (
                q_positive / localization_total if localization_total else 0.0
            )
            metrics["credit/q_total_mean"] = (
                q_total_sum / localization_total if localization_total else 0.0
            )
            metrics["credit/segment_fallback_fraction"] = (
                segment_fallbacks / segmented_responses
                if segmented_responses
                else 0.0
            )
            metrics["credit/multiplier_cap_hit_fraction"] = (
                multiplier_cap_hits / localized if localized else 0.0
            )
            known_fallbacks = 0
            for reason in (
                "construction",
                "scoring",
                "no_probability_signal",
                "direction_gate_closed",
                "ablation_failed",
            ):
                count = fallback_reasons[reason]
                known_fallbacks += count
                metrics[f"credit/fallback_reason_{reason}_count"] = float(
                    count
                )
                metrics[f"credit/fallback_reason_{reason}_fraction"] = (
                    count / localization_total if localization_total else 0.0
                )
            other_fallbacks = max(0, fallback - known_fallbacks)
            metrics["credit/fallback_reason_other_count"] = float(
                other_fallbacks
            )
            metrics["credit/fallback_reason_other_fraction"] = (
                other_fallbacks / localization_total
                if localization_total
                else 0.0
            )
            metrics["credit/max_multiplier"] = max_multiplier
            metrics["credit/channel_conservation_error_max"] = (
                max_conservation_error
            )
            for channel in self.local_channels:
                count = (
                    localized_by_channel[channel] + fallback_by_channel[channel]
                )
                metrics[f"credit/localized_fraction_{channel}"] = (
                    localized_by_channel[channel] / count if count else 0.0
                )
                metrics[f"credit/fallback_fraction_{channel}"] = (
                    fallback_by_channel[channel] / count if count else 0.0
                )
                metrics[f"credit/direction_gate_closed_fraction_{channel}"] = (
                    (count - direction_gate_by_channel[channel]) / count
                    if count
                    else 0.0
                )
                metrics[f"credit/absolute_gate_open_fraction_{channel}"] = (
                    absolute_gate_by_channel[channel] / count
                    if count
                    else 0.0
                )
                metrics[f"credit/positive_q_fraction_{channel}"] = (
                    q_positive_by_channel[channel] / count
                    if count
                    else 0.0
                )
            for channel, values in self.thresholds.values.items():
                metrics[f"credit/token_threshold_{channel}"] = values["token"]
                metrics.setdefault(
                    f"credit/token_pool_size_{channel}", 0.0
                )
                metrics.setdefault(
                    f"credit/token_threshold_update_ready_{channel}", 0.0
                )
                if (
                    self.localization_granularity
                    == SENTENCE_LOCALIZATION
                ):
                    metrics[f"credit/sentence_threshold_{channel}"] = values[
                        "sentence"
                    ]
                    metrics.setdefault(
                        f"credit/sentence_pool_size_{channel}", 0.0
                    )
                    metrics.setdefault(
                        f"credit/sentence_threshold_update_ready_{channel}",
                        0.0,
                    )

            for name in (
                "requested_count",
                "constructed_count",
                "scored_count",
                "construction_failure_count",
                "scoring_failure_count",
                "success_fraction",
                "chunk_count",
                "failed_chunk_count",
                "construction_s",
                "scoring_s",
            ):
                metrics.setdefault(f"credit/counterfactual_{name}", 0.0)
            metrics.setdefault("credit/localization_s", 0.0)

            final_errors = []
            for row in range(len(records)):
                if not valid[row]:
                    continue
                length = int(response_mask[row].sum().item())
                if length:
                    final_errors.append(
                        abs(
                            float(advantages[row, :length].mean())
                            - float(sequence_advantages[row] / rms)
                        )
                    )
            metrics["credit/final_conservation_error_max"] = max(
                final_errors, default=0.0
            )

        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
        advantages[~valid_t] = 0
        batch.batch["response_mask"][~valid_t] = 0
        advantages = advantages * batch.batch["response_mask"]
        active = batch.batch["response_mask"].bool()
        metrics["credit/final_advantage_rms"] = (
            float(torch.sqrt(advantages[active].square().mean()))
            if active.any()
            else 0.0
        )
        if trace_enabled:
            trace_started = time.monotonic()
            try:
                trace_records = build_credit_trace_records(
                    step=self.global_steps,
                    method=self.method,
                    channels=self.channels,
                    tokenizer=self.tokenizer,
                    batch=batch,
                    records=records,
                    channel_sets=channel_sets,
                    rewards=rewards,
                    valid=valid,
                    response_lengths=trace_response_lengths,
                    weights=weights,
                    advantages=advantages,
                    channel_advantages=channel_advantages,
                    scalar_rewards=scalar_rewards,
                    scalar_advantages=scalar_advantages,
                    sequence_advantages=sequence_advantages,
                    rms_scale=rms_scale,
                    segment_cache=segment_cache,
                    segment_fallback_rows=segment_fallback_rows,
                    localization_results=localization_results,
                    thresholds_used=thresholds_used,
                    localization_mode=self.localization_mode,
                    localization_granularity=self.localization_granularity,
                )
                self._credit_trace_writer.write(
                    self.global_steps, trace_records
                )
                metrics["credit/trace_failed"] = 0.0
            except Exception as exc:
                logger.warning(
                    "Credit trace failed at step %s; training continues: %s: %s",
                    self.global_steps,
                    type(exc).__name__,
                    exc,
                )
                metrics["credit/trace_failed"] = 1.0
            metrics["credit/trace_s"] = time.monotonic() - trace_started
        batch.batch["advantages"] = advantages.detach()
        batch.batch["returns"] = advantages.detach()
        self._credit_metrics = metrics
        return batch

    @staticmethod
    def _reward_extra_info(extra_fields: list[Any]) -> dict[str, np.ndarray]:
        """Flatten reward-loop annotations stored inside TQ extra_fields."""
        infos = [
            value.get("reward_extra_info", {})
            if isinstance(value, dict)
            else {}
            for value in extra_fields
        ]
        keys = {key for info in infos for key in info}
        return {
            key: np.asarray([info.get(key) for info in infos], dtype=object)
            for key in keys
        }

    def _compute_advantage(
        self,
        batch: KVBatchMeta,
        metrics: dict,
    ) -> KVBatchMeta:
        """Compute MemCalib advantages on non-padding rows."""
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError(
                "MemCalib requires algorithm.use_kl_in_reward=false; "
                "KL is included in the actor loss"
            )

        real_indices = [
            index
            for index, tag in enumerate(batch.tags)
            if not tag.get("is_padding", False)
        ]
        if not real_indices:
            raise RuntimeError("TransferQueue batch contains no real trajectories")
        real_keys = [batch.keys[index] for index in real_indices]
        fields = [
            "uid",
            "responses",
            "response_mask",
            "old_log_probs",
            "reward_model",
            "extra_fields",
        ]
        real_td = tq.kv_batch_get(
            keys=real_keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        original_response_mask = real_td["response_mask"]
        data = DataProto.from_tensordict(real_td.to_padded_tensor())
        extra_fields = data.non_tensor_batch.pop("extra_fields").tolist()
        data.non_tensor_batch.update(self._reward_extra_info(extra_fields))
        data = self._compute_credit_advantage(data)

        output = TensorDict(
            {
                "advantages": response_to_nested(
                    data.batch["advantages"], original_response_mask
                ),
                "returns": response_to_nested(
                    data.batch["returns"], original_response_mask
                ),
                "response_mask": response_to_nested(
                    data.batch["response_mask"], original_response_mask
                ),
            },
            batch_size=len(real_keys),
        )
        updated_real_batch = tq.kv_batch_put(
            keys=real_keys,
            partition_id=batch.partition_id,
            fields=output,
        )

        padding_keys = [
            key
            for key, tag in zip(batch.keys, batch.tags, strict=True)
            if tag.get("is_padding", False)
        ]
        if padding_keys:
            padding = tq.kv_batch_get(
                keys=padding_keys,
                partition_id=batch.partition_id,
                select_fields=["response_mask"],
            )
            zeros = padding["response_mask"].clone().float()
            updated_padding_batch = tq.kv_batch_put(
                keys=padding_keys,
                partition_id=batch.partition_id,
                fields=TensorDict(
                    {"advantages": zeros, "returns": zeros.clone()},
                    batch_size=len(padding_keys),
                ),
            )
            if (
                updated_real_batch.fields is not None
                and updated_padding_batch.fields is not None
                and set(updated_real_batch.fields)
                != set(updated_padding_batch.fields)
            ):
                raise RuntimeError(
                    "TransferQueue field schemas differ between real and padding "
                    "trajectories after writing MemCalib advantages"
                )

        # KVBatchMeta.fields controls which columns the worker bridge retrieves.
        # Keep the original full-batch ordering/tags and runtime extra_info, but
        # propagate the refreshed field list returned by kv_batch_put so actor
        # workers receive the newly written advantages and returns.
        batch.fields = updated_real_batch.fields

        metrics.update(self._credit_metrics)
        self._credit_metrics = {}
        return batch

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
    ) -> dict[str, float]:
        """Keep verl metrics but recompute Judge metrics on valid rows only."""
        metrics = super()._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
        )
        metrics = {
            key: value
            for key, value in metrics.items()
            if not (
                key.startswith(("val-core/", "val-aux/"))
                and any(
                    f"/{field}/" in key
                    for field in FILTERED_VALIDATION_FIELDS
                )
            )
        }
        metrics.update(
            summarize_valid_only_source_metrics(
                reward_extra_infos_dict,
                data_sources,
            )
        )
        metrics.update(
            summarize_validation_calibration(reward_extra_infos_dict)
        )
        return metrics

    def _save_checkpoint(self):
        super()._save_checkpoint()
        step_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
        )
        os.makedirs(step_dir, exist_ok=True)
        with open(
            os.path.join(step_dir, "memcalib_credit_state.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "localization_mode": self.localization_mode,
                    "localization_granularity": self.localization_granularity,
                    "thresholds": self.thresholds.state_dict(),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _load_checkpoint(self):
        super()._load_checkpoint()
        state_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "memcalib_credit_state.json",
        )
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            saved_mode = state.get("localization_mode")
            saved_granularity = state.get("localization_granularity")
            if saved_mode is not None and saved_mode != self.localization_mode:
                raise ValueError(
                    "Checkpoint localization mode does not match config: "
                    f"{saved_mode} != {self.localization_mode}"
                )
            if (
                saved_granularity is not None
                and saved_granularity != self.localization_granularity
            ):
                raise ValueError(
                    "Checkpoint localization granularity does not match config: "
                    f"{saved_granularity} != {self.localization_granularity}"
                )
            self.thresholds.load_state_dict(state["thresholds"])
