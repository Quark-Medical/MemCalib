"""Configurable channel rewards, group normalization, and localization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import torch


MERGED_CHANNELS = ("A-", "B+", "B-", "C+", "C-", "A0+", "B0-", "C0-")
SPLIT_CHANNELS = (
    "AB-",
    "AC-",
    "B+",
    "B-",
    "C+",
    "C-",
    "A0+",
    "B0-",
    "C0-",
)
MERGED_LOCAL_CHANNELS = frozenset({"A-", "B+", "B-", "C+", "C-"})
SPLIT_LOCAL_CHANNELS = frozenset(
    {"AB-", "AC-", "B+", "B-", "C+", "C-"}
)
POSITIVE_ONLY_LOCALIZATION = "positive_only"
ORDERED_BIDIRECTIONAL_LOCALIZATION = "ordered_bidirectional"
ABSOLUTE_MAGNITUDE_LOCALIZATION = "absolute_magnitude"
LOCALIZATION_MODES = frozenset(
    {
        POSITIVE_ONLY_LOCALIZATION,
        ORDERED_BIDIRECTIONAL_LOCALIZATION,
        ABSOLUTE_MAGNITUDE_LOCALIZATION,
    }
)
POSITIVE_DELTA_TRANSFORM = "positive"
NEGATIVE_DELTA_TRANSFORM = "negative"
ABSOLUTE_DELTA_TRANSFORM = "absolute"
DELTA_TRANSFORMS = frozenset(
    {
        POSITIVE_DELTA_TRANSFORM,
        NEGATIVE_DELTA_TRANSFORM,
        ABSOLUTE_DELTA_TRANSFORM,
    }
)
SENTENCE_LOCALIZATION = "sentence"
TOKEN_LOCALIZATION = "token"
LOCALIZATION_GRANULARITIES = frozenset(
    {SENTENCE_LOCALIZATION, TOKEN_LOCALIZATION}
)
ORDERED_LOCALIZATION_SIGNS = {
    "A-": 1,
    "AB-": 1,
    "AC-": 1,
    "B0-": -1,
    "B+": 1,
    "B-": 1,
    "C0-": -1,
    "C+": 1,
    "C-": -1,
}
DEFAULT_ETAS = {
    "A-": 0.7,
    "AB-": 0.7,
    "AC-": 0.7,
    "B0-": 0.5,
    "B+": 0.5,
    "B-": 0.7,
    "C0-": 0.5,
    "C+": 0.3,
    "C-": 0.5,
}


@dataclass(frozen=True)
class ChannelLayout:
    channels: tuple[str, ...]
    local_channels: frozenset[str]
    merge_a_overuse: bool


MERGED_LAYOUT = ChannelLayout(
    MERGED_CHANNELS,
    MERGED_LOCAL_CHANNELS,
    True,
)
SPLIT_LAYOUT = ChannelLayout(
    SPLIT_CHANNELS,
    SPLIT_LOCAL_CHANNELS,
    False,
)


def channel_layout(
    method: str,
    *,
    fine_grained_merge_a_overuse: bool = True,
) -> ChannelLayout:
    """Resolve the reward-channel layout for a training method."""
    if method not in {"fine_grained", "gdpo", "grpo"}:
        raise ValueError(f"Unknown credit.method={method}")
    if method == "fine_grained" and fine_grained_merge_a_overuse:
        return MERGED_LAYOUT
    return SPLIT_LAYOUT


def channel_localization_transforms(
    layout: ChannelLayout,
    *,
    mode: str = POSITIVE_ONLY_LOCALIZATION,
) -> dict[str, str]:
    """Return active channels and their full-minus-ablated delta transform."""
    if mode not in LOCALIZATION_MODES:
        choices = ", ".join(sorted(LOCALIZATION_MODES))
        raise ValueError(
            f"Unknown credit.localization.mode={mode}; expected one of {choices}"
        )
    if mode == POSITIVE_ONLY_LOCALIZATION:
        return {
            channel: POSITIVE_DELTA_TRANSFORM
            for channel in layout.channels
            if channel in layout.local_channels
        }
    if mode == ABSOLUTE_MAGNITUDE_LOCALIZATION:
        return {
            channel: ABSOLUTE_DELTA_TRANSFORM
            for channel in layout.channels
            if channel in ORDERED_LOCALIZATION_SIGNS
        }
    return {
        channel: (
            POSITIVE_DELTA_TRANSFORM
            if ORDERED_LOCALIZATION_SIGNS[channel] > 0
            else NEGATIVE_DELTA_TRANSFORM
        )
        for channel in layout.channels
        if channel in ORDERED_LOCALIZATION_SIGNS
    }


def channel_localization_signs(
    layout: ChannelLayout,
    *,
    mode: str = POSITIVE_ONLY_LOCALIZATION,
) -> dict[str, int]:
    """Compatibility view of transforms; absolute magnitude has sign zero."""
    sign_by_transform = {
        POSITIVE_DELTA_TRANSFORM: 1,
        NEGATIVE_DELTA_TRANSFORM: -1,
        ABSOLUTE_DELTA_TRANSFORM: 0,
    }
    return {
        channel: sign_by_transform[transform]
        for channel, transform in channel_localization_transforms(
            layout, mode=mode
        ).items()
    }


def effective_channel_weights(
    configured: Mapping[str, Any],
    *,
    channels: Iterable[str],
) -> dict[str, float]:
    """Return the configured post-normalization weights for active channels."""
    return {
        channel: float(configured.get(channel, 1.0)) for channel in channels
    }


def _channel(target: str, actual: str, layout: ChannelLayout) -> str:
    channels = {
        ("A", "A"): "A0+",
        ("A", "B"): "A-" if layout.merge_a_overuse else "AB-",
        ("A", "C"): "A-" if layout.merge_a_overuse else "AC-",
        ("B", "A"): "B0-",
        ("B", "B"): "B+",
        ("B", "C"): "B-",
        ("C", "A"): "C0-",
        ("C", "B"): "C-",
        ("C", "C"): "C+",
    }
    return channels[(target, actual)]


def channel_rewards(
    record: dict[str, Any],
    judgments: Iterable[dict[str, Any]],
    *,
    kappa_a: float,
    layout: ChannelLayout = MERGED_LAYOUT,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Map every atom to exactly one channel and compute normalized rewards."""
    atoms = {str(atom["atom_id"]): atom for atom in record["memories"]}
    judged = {str(item["atom_id"]): item for item in judgments}
    if set(atoms) != set(judged):
        raise ValueError("Judge atoms do not match record atoms")

    sets = {channel: [] for channel in layout.channels}
    rewards = {channel: 0.0 for channel in layout.channels}
    denominators = {
        target: max(1, sum(atom["u_star"] == target for atom in atoms.values()))
        for target in ("A", "B", "C")
    }
    for atom_id, atom in atoms.items():
        item = judged[atom_id]
        target = str(atom["u_star"])
        actual = str(item["predicted_usage_level"])
        if item.get("u_star") != target or actual not in {"A", "B", "C"}:
            raise ValueError(f"Malformed judgment for {atom_id}")
        channel = _channel(target, actual, layout)
        sets[channel].append(atom_id)
        value = 1.0 / denominators[target]
        if channel in {"A-", "AB-", "AC-", "B-", "C-", "B0-", "C0-"}:
            value = -value
        if channel in {"A-", "AC-"} and actual == "C":
            value *= kappa_a
        if channel == "C0-":
            value *= 2.0
        rewards[channel] += value
    if sum(len(values) for values in sets.values()) != len(atoms):
        raise AssertionError("Channel partition does not cover every atom")
    return sets, rewards


def group_channel_advantages(
    rewards: torch.Tensor,
    uids: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float,
    sigma_min: float,
) -> torch.Tensor:
    """Population-std GRPO normalization, separately per channel and prompt."""
    result = torch.zeros_like(rewards)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[uid].append(index)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=rewards.device)
    with torch.no_grad():
        for indices in groups.values():
            selected = torch.as_tensor(indices, device=rewards.device)
            selected = selected[valid[selected]]
            if selected.numel() <= 1:
                continue
            values = rewards[selected]
            means = values.mean(dim=0)
            stds = values.std(dim=0, unbiased=False)
            normalized = (values - means) / (stds + epsilon)
            normalized[:, stds < sigma_min] = 0
            result[selected] = normalized
    return result


def group_scalar_advantages(
    rewards: torch.Tensor,
    uids: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float,
    sigma_min: float,
) -> torch.Tensor:
    result = torch.zeros_like(rewards)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[uid].append(index)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=rewards.device)
    with torch.no_grad():
        for indices in groups.values():
            selected = torch.as_tensor(indices, device=rewards.device)
            selected = selected[valid[selected]]
            if selected.numel() <= 1:
                continue
            values = rewards[selected]
            std = values.std(unbiased=False)
            if std >= sigma_min:
                result[selected] = (values - values.mean()) / (std + epsilon)
    return result


def project_multipliers(
    raw: list[float],
    lengths: list[int],
    *,
    cap: float,
    tolerance: float = 1e-10,
) -> list[float]:
    if cap < 1 or len(raw) != len(lengths) or not raw or any(x <= 0 for x in lengths):
        raise ValueError("Invalid multiplier projection inputs")
    total = float(sum(lengths))

    def weighted_mean(tau: float) -> float:
        return sum(
            length * min(cap, max(0.0, value - tau))
            for value, length in zip(raw, lengths)
        ) / total

    low = min(value - cap for value in raw) - 1
    high = max(raw) + 1
    for _ in range(100):
        middle = (low + high) / 2
        if weighted_mean(middle) > 1:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    tau = (low + high) / 2
    result = [min(cap, max(0.0, value - tau)) for value in raw]
    if abs(sum(x * n for x, n in zip(result, lengths)) / total - 1) > 1e-7:
        raise AssertionError("Multiplier projection failed conservation")
    return result


def project_token_multipliers(
    raw: torch.Tensor,
    *,
    cap: float,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Project token multipliers efficiently while preserving their mean."""
    values = raw.detach().to(dtype=torch.float64, device="cpu")
    if cap < 1 or values.ndim != 1 or values.numel() == 0:
        raise ValueError("Invalid token multiplier projection inputs")
    low = float((values - cap).min()) - 1
    high = float(values.max()) + 1
    for _ in range(100):
        middle = (low + high) / 2
        if float((values - middle).clamp(0.0, cap).mean()) > 1:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    tau = (low + high) / 2
    result = (values - tau).clamp(0.0, cap)
    if abs(float(result.mean()) - 1) > 1e-7:
        raise AssertionError("Token multiplier projection failed conservation")
    return result


@dataclass
class LocalizationResult:
    token_advantages: torch.Tensor
    token_pool: list[float]
    sentence_pool: list[float]
    sentence_coverage: list[float]
    sentence_q: list[float]
    sentence_weights: list[float]
    sentence_multipliers: list[float]
    localized: bool
    fallback: str
    max_multiplier: float
    conservation_error: float
    direction_gate: bool
    absolute_gate: bool
    q_total: float
    localization_sign: int | None = 1
    localization_transform: str = POSITIVE_DELTA_TRANSFORM
    granularity: str = SENTENCE_LOCALIZATION
    raw_token_pool: list[float] = field(default_factory=list)


def _resolve_delta_transform(
    localization_transform: str | None,
    localization_sign: int,
) -> tuple[str, int | None]:
    if localization_transform is None:
        if localization_sign not in {-1, 1}:
            raise ValueError("localization_sign must be -1 or 1")
        transform = (
            POSITIVE_DELTA_TRANSFORM
            if localization_sign > 0
            else NEGATIVE_DELTA_TRANSFORM
        )
    else:
        transform = str(localization_transform)
        if transform not in DELTA_TRANSFORMS:
            choices = ", ".join(sorted(DELTA_TRANSFORMS))
            raise ValueError(
                f"Unknown localization_transform={transform}; "
                f"expected one of {choices}"
            )
    sign = {
        POSITIVE_DELTA_TRANSFORM: 1,
        NEGATIVE_DELTA_TRANSFORM: -1,
        ABSOLUTE_DELTA_TRANSFORM: None,
    }[transform]
    return transform, sign


def _transform_delta(
    raw_delta: torch.Tensor,
    transform: str,
) -> torch.Tensor:
    if transform == POSITIVE_DELTA_TRANSFORM:
        return raw_delta
    if transform == NEGATIVE_DELTA_TRANSFORM:
        return -raw_delta
    if transform == ABSOLUTE_DELTA_TRANSFORM:
        return raw_delta.abs()
    raise ValueError(f"Unsupported delta transform: {transform}")


def localize_advantage(
    *,
    channel: str,
    reward: float,
    advantage: float,
    full_log_probs: torch.Tensor,
    ablated_log_probs: torch.Tensor | None,
    segments: list[list[int]],
    token_threshold: float,
    sentence_threshold: float,
    absolute_token_threshold: float,
    absolute_sentence_threshold: float,
    delta_clip: float,
    coverage_gamma: float,
    eta: float,
    multiplier_cap: float,
    fallback_reason: str = "",
    localization_sign: int = 1,
    localization_transform: str | None = None,
    granularity: str = SENTENCE_LOCALIZATION,
) -> LocalizationResult:
    transform, resolved_sign = _resolve_delta_transform(
        localization_transform,
        localization_sign,
    )
    if granularity not in LOCALIZATION_GRANULARITIES:
        choices = ", ".join(sorted(LOCALIZATION_GRANULARITIES))
        raise ValueError(
            f"Unknown localization granularity={granularity}; "
            f"expected one of {choices}"
        )
    length = int(full_log_probs.numel())
    uniform = torch.full((length,), advantage, dtype=torch.float32)
    direction_gate = reward * advantage > 0
    if ablated_log_probs is None:
        return LocalizationResult(
            token_advantages=uniform,
            token_pool=[],
            sentence_pool=[],
            sentence_coverage=[],
            sentence_q=[],
            sentence_weights=[],
            sentence_multipliers=[],
            localized=False,
            fallback=fallback_reason or "ablation_failed",
            max_multiplier=1.0,
            conservation_error=0.0,
            direction_gate=direction_gate,
            absolute_gate=False,
            q_total=0.0,
            localization_sign=resolved_sign,
            localization_transform=transform,
            granularity=granularity,
        )
    if full_log_probs.shape != ablated_log_probs.shape:
        raise ValueError("Full and ablated log probabilities have different shapes")
    if [value for segment in segments for value in segment] != list(range(length)):
        raise ValueError("Segments do not cover every valid response token")

    raw_delta = (
        full_log_probs.float() - ablated_log_probs.float()
    ).cpu()
    aligned_delta = _transform_delta(raw_delta, transform)
    raw_clipped = raw_delta.clamp(-delta_clip, delta_clip)
    clipped = _transform_delta(raw_clipped, transform)

    if granularity == SENTENCE_LOCALIZATION:
        unit_indices = segments
        unit_means = [
            float(clipped[torch.as_tensor(indices)].mean())
            for indices in unit_indices
        ]
        coverage = [
            float(
                (
                    aligned_delta[torch.as_tensor(indices)] > token_threshold
                ).float().mean()
            )
            for indices in unit_indices
        ]
        absolute_gate = (
            float(aligned_delta.max()) > absolute_token_threshold
            and max(unit_means) > absolute_sentence_threshold
        )
        q_values = [
            max(0.0, mean - sentence_threshold) * (cover**coverage_gamma)
            if absolute_gate
            else 0.0
            for mean, cover in zip(unit_means, coverage)
        ]
    else:
        unit_indices = []
        unit_means = clipped.tolist()
        coverage = (
            aligned_delta > token_threshold
        ).to(dtype=torch.float32).tolist()
        absolute_gate = float(aligned_delta.max()) > absolute_token_threshold
        q_values = (
            (clipped - token_threshold).clamp_min(0).tolist()
            if absolute_gate
            else [0.0 for _ in unit_means]
        )

    q_total = sum(q_values)
    weights = (
        [value / q_total for value in q_values]
        if q_total > 0
        else [0.0 for _ in q_values]
    )
    if fallback_reason or not absolute_gate or q_total <= 0 or not direction_gate:
        reason = fallback_reason or (
            "no_probability_signal"
            if not absolute_gate or q_total <= 0
            else "direction_gate_closed"
        )
        return LocalizationResult(
            token_advantages=uniform,
            token_pool=clipped.tolist(),
            sentence_pool=unit_means,
            sentence_coverage=coverage,
            sentence_q=q_values,
            sentence_weights=weights,
            sentence_multipliers=[1.0 for _ in unit_means],
            localized=False,
            fallback=reason,
            max_multiplier=1.0,
            conservation_error=0.0,
            direction_gate=direction_gate,
            absolute_gate=absolute_gate,
            q_total=q_total,
            localization_sign=resolved_sign,
            localization_transform=transform,
            granularity=granularity,
            raw_token_pool=raw_clipped.tolist(),
        )

    if granularity == SENTENCE_LOCALIZATION:
        lengths = [len(indices) for indices in unit_indices]
        raw = [
            length / unit_length * weight
            for unit_length, weight in zip(lengths, weights)
        ]
        multipliers = project_multipliers(
            raw,
            lengths,
            cap=multiplier_cap,
        )
        token_advantages = uniform.clone()
        for indices, multiplier in zip(unit_indices, multipliers):
            value = advantage * (1 + eta * (multiplier - 1))
            token_advantages[torch.as_tensor(indices)] = value
    else:
        raw = torch.as_tensor(weights, dtype=torch.float64) * length
        multiplier_tensor = project_token_multipliers(
            raw,
            cap=multiplier_cap,
        )
        multipliers = multiplier_tensor.tolist()
        token_advantages = (
            advantage * (1 + eta * (multiplier_tensor - 1))
        ).to(dtype=torch.float32)

    conserved = float(token_advantages.mean())
    return LocalizationResult(
        token_advantages=token_advantages,
        token_pool=clipped.tolist(),
        sentence_pool=unit_means,
        sentence_coverage=coverage,
        sentence_q=q_values,
        sentence_weights=weights,
        sentence_multipliers=multipliers,
        localized=True,
        fallback="",
        max_multiplier=max(multipliers),
        conservation_error=abs(conserved - advantage),
        direction_gate=direction_gate,
        absolute_gate=absolute_gate,
        q_total=q_total,
        localization_sign=resolved_sign,
        localization_transform=transform,
        granularity=granularity,
        raw_token_pool=raw_clipped.tolist(),
    )


class AdaptiveThresholds:
    def __init__(
        self,
        *,
        initial_token: float,
        initial_sentence: float,
        channels: Iterable[str] = MERGED_LOCAL_CHANNELS,
    ):
        self.channels = tuple(channels)
        self.values = {
            channel: {"token": initial_token, "sentence": initial_sentence}
            for channel in self.channels
        }

    def update(
        self,
        token_pools: dict[str, list[float]],
        sentence_pools: dict[str, list[float]],
        *,
        token_quantile: float,
        sentence_quantile: float,
        absolute_token: float,
        absolute_sentence: float,
        ema: float,
        min_token_samples: int,
        min_sentence_samples: int,
    ) -> None:
        for channel in self.channels:
            tokens = token_pools.get(channel, [])
            sentences = sentence_pools.get(channel, [])
            if len(tokens) >= min_token_samples:
                candidate = max(absolute_token, float(np.quantile(tokens, token_quantile)))
                old = self.values[channel]["token"]
                self.values[channel]["token"] = ema * old + (1 - ema) * candidate
            if len(sentences) >= min_sentence_samples:
                candidate = max(
                    absolute_sentence,
                    float(np.quantile(sentences, sentence_quantile)),
                )
                old = self.values[channel]["sentence"]
                self.values[channel]["sentence"] = ema * old + (1 - ema) * candidate

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {channel: dict(values) for channel, values in self.values.items()}

    def load_state_dict(self, state: dict[str, dict[str, float]]) -> None:
        if set(state) != set(self.channels):
            raise ValueError("Threshold state has different channels")
        self.values = {
            channel: {
                "token": float(values["token"]),
                "sentence": float(values["sentence"]),
            }
            for channel, values in state.items()
        }
