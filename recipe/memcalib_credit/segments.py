"""Production entry point for deterministic Markdown-aware segmentation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence


def token_char_spans(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> tuple[str, list[tuple[int, int]]]:
    """Decode tokens and recover a monotonic character span for each token."""
    decoded = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    try:
        encoded = tokenizer(
            decoded,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) == list(token_ids):
            return decoded, [
                (int(start), int(end)) for start, end in encoded["offset_mapping"]
            ]
    except (NotImplementedError, TypeError, ValueError):
        pass

    prefix_lengths = [0]
    for end in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(
            list(token_ids[:end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix_lengths.append(len(prefix))
    if prefix_lengths[-1] != len(decoded) or any(
        right < left for left, right in zip(prefix_lengths, prefix_lengths[1:])
    ):
        raise ValueError("Could not align generated tokens to decoded response")
    return decoded, list(zip(prefix_lengths, prefix_lengths[1:]))


@lru_cache(maxsize=1)
def _segmentation_context() -> tuple[Any, Any, Any]:
    """Load the parser once per worker process, on first segmentation use."""
    try:
        import spacy
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError(
            "MemCalib segmentation requires spaCy and markdown-it-py; "
            "install recipe/memcalib_credit/requirements.txt"
        ) from exc

    try:
        nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "parser"])
    except OSError as exc:
        raise RuntimeError(
            "MemCalib segmentation requires the en_core_web_sm spaCy model; "
            "install recipe/memcalib_credit/requirements.txt"
        ) from exc

    from recipe.memcalib_credit import _segments_policy

    markdown = MarkdownIt("commonmark").enable("table")
    return _segments_policy, markdown, nlp


def segment_token_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[list[int]]:
    """Split one response into ordered, exactly covering token-index groups."""
    if not token_ids:
        return []
    policy, markdown, nlp = _segmentation_context()
    segments, _, _, _ = policy.segment_token_ids(
        tokenizer,
        token_ids,
        md=markdown,
        nlp=nlp,
        align_tokens=token_char_spans,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    return segments
