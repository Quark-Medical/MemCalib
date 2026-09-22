"""Async Judge reward manager for verl 0.8's reward-loop API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from omegaconf import OmegaConf
from openai import AsyncOpenAI

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from recipe.memcalib_credit.credit import channel_layout, channel_rewards
from recipe.memcalib_credit.judge import (
    JudgeRequestError,
    get_rate_limiter,
    judge_one,
)
from recipe.memcalib_credit.metrics import sample_level_calibration_metrics


logger = logging.getLogger(__name__)


def _valid_response_length(data_item: Any, response_ids: Any) -> int:
    """Resolve the generated response length across reward-loop contracts."""
    if "response_mask" in data_item.batch.keys():
        valid_length = int(data_item.batch["response_mask"].sum().item())
    elif "response_len" in data_item.non_tensor_batch:
        raw_length = data_item.non_tensor_batch["response_len"]
        if hasattr(raw_length, "item"):
            raw_length = raw_length.item()
        valid_length = int(raw_length)
    else:
        raise KeyError(
            "Reward input contains neither response_mask nor response_len"
        )

    response_width = int(response_ids.shape[-1])
    if not 0 <= valid_length <= response_width:
        raise ValueError(
            "Reward response length is outside the response tensor: "
            f"length={valid_length}, width={response_width}"
        )
    return valid_length


class MemCalibRewardManager(RewardManagerBase):
    """Return one scalar reward plus the annotations used by credit assignment."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer, compute_score)
        del reward_router_address, reward_model_tokenizer

        credit = OmegaConf.to_container(config.get("credit", {}), resolve=True)
        judge = dict(credit.get("judge", {}))
        self.api_key_env = str(judge.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.base_url = str(
            judge.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        )
        self.model = str(judge.get("judge_model", "deepseek-v4-pro"))
        self.max_concurrent = int(judge.get("max_concurrent", 32))
        self.requests_per_minute = int(judge.get("requests_per_minute", 240))
        self.max_retries = int(judge.get("max_retries", 4))
        self.max_tokens = int(judge.get("max_tokens", 16384))
        self.timeout = float(judge.get("timeout", 300))
        self.kappa_a = float(credit.get("kappa_a", 2.0))
        self.method = str(credit.get("method", "fine_grained"))
        self.layout = channel_layout(
            self.method,
            fine_grained_merge_a_overuse=bool(
                credit.get("fine_grained", {}).get("merge_a_overuse", True)
            ),
        )
        configured_weights = credit.get("channel_weights", {})
        self.channel_weights = {
            channel: float(configured_weights.get(channel, 1.0))
            for channel in self.layout.channels
        }

        # RewardLoopWorker calls all run_single coroutines on one event loop.  Keeping
        # one worker in the recipe therefore shares process-wide RPM and
        # concurrency limits while still evaluating a rollout batch concurrently.
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._limiter = None

    def _ensure_runtime(self) -> tuple[AsyncOpenAI, asyncio.Semaphore, Any]:
        loop = asyncio.get_running_loop()
        if self._runtime_loop is not None and self._runtime_loop is not loop:
            raise RuntimeError("MemCalibRewardManager moved between event loops")
        self._runtime_loop = loop
        if self._client is None:
            api_key = os.environ.get(self.api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"missing_environment_variable:{self.api_key_env}"
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
            self._limiter = get_rate_limiter(self.requests_per_minute)
        return self._client, self._semaphore, self._limiter

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Score the last sequence and return reward metadata."""
        started = time.monotonic()
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        valid_length = _valid_response_length(data_item, response_ids)
        valid_response_ids = response_ids[:valid_length]
        response = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        record = (
            json.loads(ground_truth)
            if isinstance(ground_truth, str)
            else ground_truth
        )

        if valid_length <= 0 or not response.strip():
            result = self._failure("empty_response")
        else:
            try:
                client, semaphore, limiter = self._ensure_runtime()
                parsed = await judge_one(
                    client=client,
                    record=record,
                    response=response,
                    model=self.model,
                    semaphore=semaphore,
                    rate_limiter=limiter,
                    max_retries=self.max_retries,
                    max_tokens=self.max_tokens,
                )
                request_metrics = self._request_metrics(
                    parsed.pop("_request_stats", {})
                )
                judgments = parsed["judgments"]
                _, rewards = channel_rewards(
                    record,
                    judgments,
                    kappa_a=self.kappa_a,
                    layout=self.layout,
                )
                result = {
                    "memcalib_reward_valid": True,
                    "memcalib_judgments": json.dumps(
                        judgments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "memcalib_judge_error": "",
                    **self._usage_metrics(record, judgments, rewards),
                    **request_metrics,
                }
            except Exception as exc:
                logger.warning(
                    "Judge failed for record %s: %s: %s",
                    record.get("id"),
                    type(exc).__name__,
                    exc,
                )
                stats = exc.stats if isinstance(exc, JudgeRequestError) else {}
                cause = exc.__cause__ if isinstance(exc, JudgeRequestError) else exc
                cause = cause or exc
                result = self._failure(
                    f"{type(cause).__name__}:{cause}",
                    stats=stats,
                )

        result["memcalib_judge_batch_wall_s"] = time.monotonic() - started
        reward = float(result["acc"]) if result["memcalib_reward_valid"] else 0.0
        return {"reward_score": reward, "reward_extra_info": result}

    def _failure(
        self,
        error: str,
        *,
        stats: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "memcalib_reward_valid": False,
            "memcalib_judgments": "",
            "memcalib_judge_error": error[:500],
            **self._empty_metrics(),
            **self._request_metrics(stats or {}),
        }

    @staticmethod
    def _request_metrics(stats: dict[str, float]) -> dict[str, float]:
        return {
            f"memcalib_judge_{name}": float(stats.get(name, 0.0))
            for name in (
                "attempts",
                "retries",
                "latency_s",
                "rate_limit_wait_s",
                "concurrency_wait_s",
                "api_s",
                "retry_backoff_s",
            )
        }

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "acc": 0.0,
            "memcalib_overuse_rate": 0.0,
            "memcalib_underuse_rate": 0.0,
            "memcalib_usage_distance": 0.0,
            "memcalib_raw_reward": 0.0,
            "memcalib_judge_valid": 0.0,
            "memcalib_over_budget": 0.0,
            "memcalib_under_budget": 0.0,
            "memcalib_total_budget": 0.0,
            "memcalib_scs_rho_0_5": 0.0,
            "memcalib_sopb_rho_0_5": 0.0,
            "memcalib_supb_rho_0_5": 0.0,
            "memcalib_any_opb": 0.0,
            "memcalib_any_upb": 0.0,
            "memcalib_exact": 0.0,
        }

    def _usage_metrics(
        self,
        record: dict[str, Any],
        judgments: list[dict[str, Any]],
        rewards: dict[str, float],
    ) -> dict[str, float]:
        levels = {"A": 0, "B": 1, "C": 2}
        targets = {
            str(atom["atom_id"]): str(atom["u_star"])
            for atom in record["memories"]
        }
        distances = [
            levels[str(item["predicted_usage_level"])]
            - levels[targets[str(item["atom_id"])]]
            for item in judgments
        ]
        count = len(distances)
        if not count:
            raise ValueError("Record contains no atoms")
        return {
            "acc": sum(distance == 0 for distance in distances) / count,
            "memcalib_overuse_rate": sum(
                distance > 0 for distance in distances
            )
            / count,
            "memcalib_underuse_rate": sum(
                distance < 0 for distance in distances
            )
            / count,
            "memcalib_usage_distance": sum(map(abs, distances)) / count,
            "memcalib_raw_reward": sum(
                self.channel_weights[channel] * rewards[channel]
                for channel in self.layout.channels
            ),
            "memcalib_judge_valid": 1.0,
            **sample_level_calibration_metrics(distances),
        }
