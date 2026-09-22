"""Training telemetry helpers kept separate from the credit algorithm."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from verl import DataProto


_VALIDATION_CALIBRATION_FIELDS = {
    "acc": "atom_accuracy",
    "memcalib_scs_rho_0_5": "scs_rho_0_5",
    "memcalib_sopb_rho_0_5": "sopb_rho_0_5",
    "memcalib_supb_rho_0_5": "supb_rho_0_5",
    "memcalib_any_opb": "any_opb",
    "memcalib_any_upb": "any_upb",
    "memcalib_exact": "exact",
    "memcalib_over_budget": "mean_over_budget",
    "memcalib_under_budget": "mean_under_budget",
    "memcalib_total_budget": "mean_total_budget",
}

VALID_ONLY_VALIDATION_FIELDS = (
    "acc",
    "memcalib_overuse_rate",
    "memcalib_underuse_rate",
    "memcalib_usage_distance",
    "memcalib_raw_reward",
)

SAMPLE_LEVEL_VALIDATION_FIELDS = (
    "memcalib_over_budget",
    "memcalib_under_budget",
    "memcalib_total_budget",
    "memcalib_scs_rho_0_5",
    "memcalib_sopb_rho_0_5",
    "memcalib_supb_rho_0_5",
    "memcalib_any_opb",
    "memcalib_any_upb",
    "memcalib_exact",
)

FILTERED_VALIDATION_FIELDS = (
    *VALID_ONLY_VALIDATION_FIELDS,
    *SAMPLE_LEVEL_VALIDATION_FIELDS,
)


def sample_level_calibration_metrics(
    distances: Iterable[int],
) -> dict[str, float]:
    """Compute primary sample-level MemCalib metrics at rho=0.5."""
    values = [int(value) for value in distances]
    if not values:
        raise ValueError("sample-level calibration requires at least one atom")

    over_budget = sum(max(value, 0) for value in values)
    under_budget = sum(max(-value, 0) for value in values)
    total_budget = over_budget + under_budget
    rho = 0.5
    return {
        "memcalib_over_budget": float(over_budget),
        "memcalib_under_budget": float(under_budget),
        "memcalib_total_budget": float(total_budget),
        "memcalib_scs_rho_0_5": rho**total_budget,
        "memcalib_sopb_rho_0_5": 1 - rho**over_budget,
        "memcalib_supb_rho_0_5": 1 - rho**under_budget,
        "memcalib_any_opb": float(over_budget > 0),
        "memcalib_any_upb": float(under_budget > 0),
        "memcalib_exact": float(total_budget == 0),
    }


def summarize_validation_calibration(
    extra: Mapping[str, Sequence[Any]],
    *,
    condition: str = "full_memory",
) -> dict[str, float]:
    """Aggregate only valid Judge rows into TensorBoard-ready metrics."""
    valid = np.asarray(extra.get("memcalib_reward_valid", []), dtype=bool)
    expected = len(valid)
    if not expected:
        raise ValueError("validation produced no Judge rows")

    valid_count = int(valid.sum())
    if not valid_count:
        raise ValueError("validation produced no valid Judge rows")

    result = {
        "val-aux/memcalib/coverage/expected_samples": float(expected),
        "val-aux/memcalib/coverage/samples": float(valid_count),
        "val-aux/memcalib/coverage/invalid_samples": float(
            expected - valid_count
        ),
        "val-aux/memcalib/coverage/sample_fraction": valid_count / expected,
    }
    for source, label in _VALIDATION_CALIBRATION_FIELDS.items():
        values = np.asarray(extra.get(source, []), dtype=float)
        if len(values) != expected:
            raise ValueError(
                f"validation metric {source} length {len(values)} != {expected}"
            )
        selected = values[valid]
        if not np.isfinite(selected).all():
            raise ValueError(f"validation metric {source} contains non-finite values")
        result[f"val-core/memcalib/{condition}/{label}"] = float(
            selected.mean()
        )
    return result


def summarize_valid_only_source_metrics(
    extra: Mapping[str, Sequence[Any]],
    data_sources: Sequence[Any],
) -> dict[str, float]:
    """Recompute validation quality metrics without invalid Judge rows."""
    valid = np.asarray(extra.get("memcalib_reward_valid", []), dtype=bool)
    sources = np.asarray(data_sources, dtype=str)
    expected = len(valid)
    if len(sources) != expected:
        raise ValueError("validation validity and source lengths differ")

    result = {}
    for field in VALID_ONLY_VALIDATION_FIELDS:
        if field not in extra:
            continue
        values = np.asarray(extra[field], dtype=float)
        if len(values) != expected:
            raise ValueError(
                f"validation metric {field} length {len(values)} != {expected}"
            )
        section = "val-core" if field == "acc" else "val-aux"
        for source in sorted(set(sources)):
            selected = valid & (sources == source)
            if selected.any():
                result[f"{section}/{source}/{field}/mean@1"] = float(
                    values[selected].mean()
                )
    return result


def _extra_array(batch: DataProto, key: str, length: int) -> np.ndarray:
    return np.asarray(
        batch.non_tensor_batch.get(key, np.zeros(length)),
        dtype=float,
    )


def add_judge_metrics(
    metrics: dict[str, float],
    batch: DataProto,
    valid: np.ndarray,
) -> None:
    length = len(valid)
    fields = (
        "attempts",
        "retries",
        "latency_s",
        "rate_limit_wait_s",
        "concurrency_wait_s",
        "api_s",
        "retry_backoff_s",
    )
    for name in fields:
        values = _extra_array(batch, f"memcalib_judge_{name}", length)
        metrics[f"judge/{name}_mean"] = float(values.mean()) if length else 0.0
        metrics[f"judge/{name}_max"] = float(values.max()) if length else 0.0
        if name in {
            "retries",
            "rate_limit_wait_s",
            "concurrency_wait_s",
            "retry_backoff_s",
        }:
            metrics[f"judge/{name}_total"] = float(values.sum())
        if name == "latency_s":
            metrics["judge/latency_s_p95"] = (
                float(np.quantile(values, 0.95)) if length else 0.0
            )

    batch_wall = _extra_array(batch, "memcalib_judge_batch_wall_s", length)
    retries = _extra_array(batch, "memcalib_judge_retries", length)
    metrics["judge/batch_wall_s"] = (
        float(batch_wall.max()) if length else 0.0
    )
    metrics["judge/retried_request_fraction"] = (
        float((retries > 0).mean()) if length else 0.0
    )
    metrics["judge/valid_fraction"] = float(valid.mean()) if length else 0.0

    errors = np.asarray(
        batch.non_tensor_batch.get(
            "memcalib_judge_error", np.full(length, "")
        ),
        dtype=str,
    )
    categories = defaultdict(int)
    for keep, error in zip(valid, errors, strict=True):
        if not keep:
            categories[_judge_failure_category(error)] += 1
    for category in (
        "missing_api_key",
        "empty_response",
        "timeout",
        "rate_limit",
        "parse",
        "api",
        "reward_input",
        "other",
    ):
        count = categories[category]
        metrics[f"judge/failure_{category}_count"] = float(count)
        metrics[f"judge/failure_{category}_fraction"] = (
            count / length if length else 0.0
        )


def _judge_failure_category(error: str) -> str:
    value = error.lower()
    if not value:
        return "reward_input"
    if "missing_environment_variable" in value:
        return "missing_api_key"
    if "empty_response" in value:
        return "empty_response"
    if "timeout" in value:
        return "timeout"
    if any(
        term in value
        for term in ("ratelimit", "rate_limit", "429", "too many requests")
    ):
        return "rate_limit"
    if any(term in value for term in ("valueerror", "json", "atom")):
        return "parse"
    if any(term in value for term in ("api", "connection", "server")):
        return "api"
    return "other"


def add_usage_metrics(
    metrics: dict[str, float],
    batch: DataProto,
    records: list[dict[str, Any]],
    valid: np.ndarray,
) -> None:
    fields = {
        "acc": "usage_accuracy",
        "memcalib_overuse_rate": "overuse_rate",
        "memcalib_underuse_rate": "underuse_rate",
        "memcalib_usage_distance": "usage_distance",
        "memcalib_raw_reward": "raw_reward",
    }
    domains = np.asarray([str(record["domain"]) for record in records])
    for source, label in fields.items():
        values = _extra_array(batch, source, len(records))
        metrics[f"credit/{label}"] = (
            float(values[valid].mean()) if valid.any() else 0.0
        )
        for domain in ("health_seed", "general"):
            selected = valid & (domains == domain)
            metrics[f"credit/{label}_{domain}"] = (
                float(values[selected].mean()) if selected.any() else 0.0
            )


def _groups(uids: np.ndarray) -> dict[Any, list[int]]:
    result: dict[Any, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        result[uid].append(index)
    return result


def add_group_metrics(
    metrics: dict[str, float],
    rewards: torch.Tensor,
    uids: np.ndarray,
    valid: np.ndarray,
    sigma_min: float,
    channels: tuple[str, ...],
) -> None:
    groups = _groups(uids)
    counts = [
        sum(bool(valid[index]) for index in indices)
        for indices in groups.values()
    ]
    group_count = len(counts)
    metrics["credit/group_count"] = float(group_count)
    metrics["credit/group_valid_response_mean"] = (
        float(np.mean(counts)) if counts else 0.0
    )
    metrics["credit/group_valid_response_min"] = (
        float(min(counts)) if counts else 0.0
    )
    metrics["credit/group_usable_fraction"] = (
        sum(count >= 2 for count in counts) / group_count
        if group_count
        else 0.0
    )
    metrics["credit/group_all_invalid_fraction"] = (
        sum(count == 0 for count in counts) / group_count
        if group_count
        else 0.0
    )

    reward_values = rewards.detach().cpu().numpy()
    usable = [
        [index for index in indices if valid[index]]
        for indices in groups.values()
        if sum(bool(valid[index]) for index in indices) >= 2
    ]
    for channel_index, channel in enumerate(channels):
        metrics[f"credit/nondegenerate_group_fraction_{channel}"] = (
            sum(
                np.std(reward_values[indices, channel_index]) >= sigma_min
                for indices in usable
            )
            / len(usable)
            if usable
            else 0.0
        )


def scalar_group_fraction(
    values: torch.Tensor,
    uids: np.ndarray,
    valid: np.ndarray,
    sigma_min: float,
) -> float:
    values_np = values.detach().cpu().numpy()
    usable = [
        [index for index in indices if valid[index]]
        for indices in _groups(uids).values()
        if sum(bool(valid[index]) for index in indices) >= 2
    ]
    return (
        sum(np.std(values_np[indices]) >= sigma_min for indices in usable)
        / len(usable)
        if usable
        else 0.0
    )
