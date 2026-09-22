"""Bailian ordered-usage Judge with bounded concurrency and request rate."""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time
from collections import deque
from typing import Any


SYSTEM_PROMPT = """You are evaluating the realized, observable footprint of each hidden atomic memory in a model response. Judge every atom independently. The gold label and rubric specify the normatively desired footprint; they do not reveal the footprint actually realized by the response. Classify actual usage before comparing it with u_star.

Usage means observable incorporation of atom-specific content, regardless of whether the response accepts, rejects, corrects, warns about, or argues against that content. Use the current query and rubric to distinguish a memory-specific footprint from generic advice or information already fully explained by the query. Ask how much of the response would change if the atomic content were removed while the current query stayed fixed. Do not infer hidden intent.

Predicted usage levels are ordered by response footprint:
- A: no observable atom-specific footprint. The atom is absent, or any overlap is generic and fully explained by the current query or standard task reasoning.
- B: bounded footprint. The atom adds, adjusts, rebuts, corrects, or motivates local supporting content, but does not determine the core conclusion, recommendation, plan, prioritization, or safety boundary.
- C: controlling footprint. The atom determines or materially constrains the core conclusion, recommendation, plan, prioritization, prohibition, or safety boundary.

Critical boundary rules:
- Acceptance is not required for usage. Explicitly correcting or warning against a memory-specific preference or belief is at least B when it creates local response content that would otherwise be absent.
- A brief local correction is B. Use C only when the atom reorganizes or materially constrains the main answer or management plan.
- Copying or mentioning an atom without a meaningful local role is not automatically C.

Return exactly one JSON object with this schema:
{
  "atom_judgments": [
    {
      "atom_id": "string",
      "u_star": "A|B|C",
      "predicted_usage_level": "A|B|C",
      "evidence_quote": "exact quote from MODEL RESPONSE, or empty when predicted level is A",
      "reason": "concise explanation of the observable footprint and its scope"
    }
  ]
}

Return every requested atom exactly once and no additional atoms. Copy u_star exactly. Return JSON only, without Markdown fences."""


class SlidingWindowLimiter:
    """Process-wide RPM limiter that remains valid across asyncio event loops."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.limit = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    async def acquire(self) -> float:
        waited = 0.0
        while True:
            now = time.monotonic()
            with self._lock:
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(now)
                    return waited
                wait = max(0.01, 60 - (now - self._timestamps[0]))
            started = time.monotonic()
            await asyncio.sleep(wait)
            waited += time.monotonic() - started


class JudgeRequestError(RuntimeError):
    """Preserve request telemetry when all Judge attempts fail."""

    def __init__(self, error: Exception, stats: dict[str, float]):
        super().__init__(str(error))
        self.stats = stats


_LIMITERS: dict[int, SlidingWindowLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(requests_per_minute: int) -> SlidingWindowLimiter:
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(requests_per_minute)
        if limiter is None:
            limiter = SlidingWindowLimiter(requests_per_minute)
            _LIMITERS[requests_per_minute] = limiter
        return limiter


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge returned no JSON object")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge output is not a JSON object")
    return value


def judge_messages(record: dict[str, Any], response: str) -> list[dict[str, str]]:
    payload = {
        "current_query": record["question"],
        "model_facing_memory": [
            {
                "parent_memory_id": block["parent_memory_id"],
                "memory_text": block["memory_text"],
            }
            for block in record["memory_blocks"]
        ],
        "model_response": response,
        "atomic_rubrics": [
            {
                "atom_id": atom["atom_id"],
                "parent_memory_id": atom["parent_memory_id"],
                "text": atom["text"],
                "u_star": atom["u_star"],
                "usage_rubric": atom["usage_rubric"],
            }
            for atom in record["memories"]
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_judgment(record: dict[str, Any], text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    raw = payload.get("atom_judgments")
    if not isinstance(raw, list):
        raise ValueError("Judge atom_judgments must be a list")

    atoms = {str(atom["atom_id"]): atom for atom in record["memories"]}
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Judge returned a non-object atom judgment")
        atom_id = str(item.get("atom_id", ""))
        if atom_id not in atoms or atom_id in by_id:
            raise ValueError(f"Unknown or repeated Judge atom_id={atom_id!r}")
        target = str(atoms[atom_id]["u_star"])
        actual = item.get("predicted_usage_level")
        if item.get("u_star") != target:
            raise ValueError(f"Judge copied wrong u_star for {atom_id}")
        if not isinstance(actual, str) or actual not in {"A", "B", "C"}:
            raise ValueError(f"Invalid actual usage for {atom_id}")
        evidence_quote = item.get("evidence_quote")
        reason = item.get("reason")
        if not isinstance(evidence_quote, str) or not isinstance(reason, str):
            raise ValueError(f"Invalid evidence_quote or reason for {atom_id}")
        by_id[atom_id] = {
            "atom_id": atom_id,
            "u_star": target,
            "predicted_usage_level": actual,
            "evidence_quote": evidence_quote,
            "reason": reason,
        }
    if set(by_id) != set(atoms):
        raise ValueError("Judge did not return every atom exactly once")

    return {
        "judgments": [by_id[str(atom["atom_id"])] for atom in record["memories"]],
    }


async def judge_one(
    *,
    client: Any,
    record: dict[str, Any],
    response: str,
    model: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: SlidingWindowLimiter,
    max_retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.monotonic()
    last_error: Exception | None = None
    attempts = 0
    rate_limit_wait = 0.0
    concurrency_wait = 0.0
    api_time = 0.0
    retry_backoff = 0.0

    def request_stats() -> dict[str, float]:
        return {
            "attempts": float(attempts),
            "retries": float(max(0, attempts - 1)),
            "latency_s": time.monotonic() - started,
            "rate_limit_wait_s": rate_limit_wait,
            "concurrency_wait_s": concurrency_wait,
            "api_s": api_time,
            "retry_backoff_s": retry_backoff,
        }

    for attempt in range(max_retries):
        attempts += 1
        try:
            rate_limit_wait += await rate_limiter.acquire()
            semaphore_started = time.monotonic()
            async with semaphore:
                concurrency_wait += time.monotonic() - semaphore_started
                api_started = time.monotonic()
                try:
                    result = await client.chat.completions.create(
                        model=model,
                        messages=judge_messages(record, response),
                        temperature=0,
                        max_tokens=max_tokens,
                        extra_body={"enable_thinking": False},
                    )
                finally:
                    api_time += time.monotonic() - api_started
            content = result.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Judge returned empty content")
            parsed = parse_judgment(record, content)
            parsed["_request_stats"] = request_stats()
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                delay = min(30.0, 1.5 * (2**attempt)) + random.uniform(0, 0.5)
                backoff_started = time.monotonic()
                await asyncio.sleep(delay)
                retry_backoff += time.monotonic() - backoff_started
    assert last_error is not None
    raise JudgeRequestError(last_error, request_stats()) from last_error
