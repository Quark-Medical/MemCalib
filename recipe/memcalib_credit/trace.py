"""Sentence-level credit traces for later analysis."""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)


class CreditTraceWriter:
    """Atomically write one lightweight gzip JSONL file per training step."""

    def __init__(self, output_dir: str | Path, *, compression_level: int = 1):
        self.output_dir = Path(output_dir).expanduser()
        self.compression_level = int(compression_level)
        if not 0 <= self.compression_level <= 9:
            raise ValueError("credit.trace.compression_level must be in [0, 9]")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, step: int, records: Iterable[Mapping[str, Any]]) -> None:
        path = self.output_dir / f"{int(step)}.jsonl.gz"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    fileobj=raw,
                    mode="wb",
                    filename="",
                    compresslevel=self.compression_level,
                    mtime=0,
                ) as compressed:
                    for record in records:
                        line = json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                        compressed.write(line + b"\n")
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def _batch_value(batch: Any, key: str, row: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None or row >= len(values):
        return default
    value = values[row]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_segments(
    tokenizer: Any,
    response_ids: torch.Tensor,
    row_segments: Mapping[int, list[list[int]]],
) -> dict[tuple[int, int], str]:
    inputs: list[list[int]] = []
    keys: list[tuple[int, int]] = []
    for row, segments in row_segments.items():
        for segment_index, indices in enumerate(segments):
            inputs.append([int(response_ids[row, index]) for index in indices])
            keys.append((row, segment_index))
    if not inputs:
        return {}
    try:
        decoded = tokenizer.batch_decode(
            inputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        logger.warning(
            "Credit trace sentence decoding failed; token spans are retained "
            "without text: %s: %s",
            type(exc).__name__,
            exc,
        )
        return {}
    return dict(zip(keys, decoded, strict=True))


def summarize_localization(
    result: Any, segments: Sequence[list[int]]
) -> dict[str, Any]:
    """Drop token-sized pools and retain only sentence-sized trace values."""
    def aggregate(values, reducer):
        if not values:
            return []
        return [
            reducer([float(values[index]) for index in indices])
            for indices in segments
        ]

    def mean(values):
        return sum(values) / len(values)

    localized_advantages = [
        float(
            result.token_advantages[torch.as_tensor(indices)].mean()
        )
        for indices in segments
    ]
    raw_sentence_pool = aggregate(result.raw_token_pool, mean)
    if result.granularity == "token":
        sentence_pool = aggregate(result.token_pool, mean)
        sentence_coverage = aggregate(result.sentence_coverage, mean)
        sentence_q = aggregate(result.sentence_q, sum)
        sentence_weights = aggregate(result.sentence_weights, sum)
        sentence_multipliers = aggregate(result.sentence_multipliers, mean)
    else:
        sentence_pool = list(result.sentence_pool)
        sentence_coverage = list(result.sentence_coverage)
        sentence_q = list(result.sentence_q)
        sentence_weights = list(result.sentence_weights)
        sentence_multipliers = list(result.sentence_multipliers)
    return {
        "localized": bool(result.localized),
        "fallback": str(result.fallback),
        "direction_gate": bool(result.direction_gate),
        "absolute_gate": bool(result.absolute_gate),
        "localization_transform": str(result.localization_transform),
        "localization_sign": (
            int(result.localization_sign)
            if result.localization_sign is not None
            else None
        ),
        "granularity": str(result.granularity),
        "q_total": float(result.q_total),
        "conservation_error": float(result.conservation_error),
        "sentence_pool": sentence_pool,
        "raw_sentence_pool": raw_sentence_pool,
        "sentence_coverage": sentence_coverage,
        "sentence_q": sentence_q,
        "sentence_weights": sentence_weights,
        "sentence_multipliers": sentence_multipliers,
        "localized_advantages": localized_advantages,
    }


def build_credit_trace_records(
    *,
    step: int,
    method: str,
    channels: Sequence[str],
    tokenizer: Any,
    batch: Any,
    records: list[dict[str, Any]],
    channel_sets: list[dict[str, list[str]]],
    rewards: torch.Tensor,
    valid: np.ndarray,
    response_lengths: list[int],
    weights: torch.Tensor,
    advantages: torch.Tensor,
    channel_advantages: torch.Tensor | None,
    scalar_rewards: torch.Tensor | None,
    scalar_advantages: torch.Tensor | None,
    sequence_advantages: torch.Tensor | None,
    rms_scale: float,
    segment_cache: Mapping[int, list[list[int]]],
    segment_fallback_rows: Mapping[int, bool],
    localization_results: Mapping[tuple[int, str], Mapping[str, Any]],
    thresholds_used: Mapping[str, Mapping[str, float]],
    localization_mode: str | None = None,
    localization_granularity: str | None = None,
) -> list[dict[str, Any]]:
    """Build trace rows exclusively from already-computed detached values."""
    response_ids = batch.batch["responses"].detach().cpu()
    reward_values = rewards.detach().float().cpu()
    weight_values = weights.detach().float().cpu()
    final_values = advantages.detach().float().cpu()
    channel_advantage_values = (
        channel_advantages.detach().float().cpu()
        if channel_advantages is not None
        else None
    )
    scalar_reward_values = (
        scalar_rewards.detach().float().cpu()
        if scalar_rewards is not None
        else None
    )
    scalar_advantage_values = (
        scalar_advantages.detach().float().cpu()
        if scalar_advantages is not None
        else None
    )
    sequence_advantage_values = (
        sequence_advantages.detach().float().cpu()
        if sequence_advantages is not None
        else None
    )

    row_segments = {}
    for row, length in enumerate(response_lengths):
        segments = segment_cache.get(row)
        row_segments[row] = (
            segments
            if segments is not None
            else ([list(range(length))] if length else [])
        )
    segment_texts = _decode_segments(tokenizer, response_ids, row_segments)

    output = []
    for row, record in enumerate(records):
        length = response_lengths[row]
        segments = row_segments[row]
        sentence_records = []
        for segment_index, indices in enumerate(segments):
            values = final_values[row, torch.as_tensor(indices)]
            sentence = {
                "segment_index": segment_index,
                "token_start": int(indices[0]),
                "token_end": int(indices[-1]) + 1,
                "token_count": len(indices),
                "final_advantage": float(values.mean()),
            }
            text = segment_texts.get((row, segment_index))
            if text is not None:
                sentence["text"] = text
            sentence_records.append(sentence)

        channel_records = {}
        for channel_index, channel in enumerate(channels):
            normalized_advantage = (
                float(channel_advantage_values[row, channel_index])
                if channel_advantage_values is not None
                else None
            )
            channel_record: dict[str, Any] = {
                "atom_ids": list(channel_sets[row][channel]),
                "reward": float(reward_values[row, channel_index]),
                "normalized_advantage": normalized_advantage,
                "channel_weight": float(weight_values[channel_index]),
            }
            result = localization_results.get((row, channel))
            if result is not None:
                localized_sentences = []
                for segment_index, _ in enumerate(segments):
                    def item(values, default=None):
                        return (
                            values[segment_index]
                            if segment_index < len(values)
                            else default
                        )

                    localized_sentences.append(
                        {
                            "segment_index": segment_index,
                            "mean_delta": item(result["sentence_pool"]),
                            "mean_raw_delta": item(
                                result["raw_sentence_pool"]
                            ),
                            "coverage": item(result["sentence_coverage"]),
                            "q": item(result["sentence_q"]),
                            "q_weight": item(result["sentence_weights"]),
                            "multiplier": item(result["sentence_multipliers"]),
                            "localized_advantage": item(
                                result["localized_advantages"]
                            ),
                        }
                    )
                channel_record["localization"] = {
                    "localized": result["localized"],
                    "fallback": result["fallback"],
                    "direction_gate": result["direction_gate"],
                    "absolute_gate": result["absolute_gate"],
                    "localization_transform": result[
                        "localization_transform"
                    ],
                    "localization_sign": result["localization_sign"],
                    "granularity": result["granularity"],
                    "token_threshold": thresholds_used.get(channel, {}).get(
                        "token"
                    ),
                    "sentence_threshold": (
                        thresholds_used.get(channel, {}).get("sentence")
                        if result["granularity"] == "sentence"
                        else None
                    ),
                    "q_total": result["q_total"],
                    "conservation_error": result["conservation_error"],
                    "sentences": localized_sentences,
                }
            channel_records[channel] = channel_record

        final_mean = float(final_values[row, :length].mean()) if length else 0.0
        expected_mean = (
            float(sequence_advantage_values[row]) / rms_scale
            if sequence_advantage_values is not None
            else final_mean
        )
        output.append(
            {
                "step": int(step),
                "row_index": row,
                "sample_id": str(record.get("id", row)),
                "uid": str(_batch_value(batch, "uid", row, "")),
                "method": method,
                "localization_mode": localization_mode,
                "localization_granularity": localization_granularity,
                "valid_for_update": bool(valid[row]),
                "response_token_count": length,
                "segment_fallback": bool(segment_fallback_rows.get(row, False)),
                "segments": sentence_records,
                "channels": channel_records,
                "scalar_reward": (
                    float(scalar_reward_values[row])
                    if scalar_reward_values is not None
                    else None
                ),
                "scalar_advantage": (
                    float(scalar_advantage_values[row])
                    if scalar_advantage_values is not None
                    else None
                ),
                "sequence_advantage": (
                    float(sequence_advantage_values[row])
                    if sequence_advantage_values is not None
                    else None
                ),
                "rms_scale": float(rms_scale),
                "final_advantage_mean": final_mean,
                "final_conservation_error": (
                    abs(final_mean - expected_mean) if valid[row] else 0.0
                ),
            }
        )
    return output
