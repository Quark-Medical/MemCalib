from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from markdown_it import MarkdownIt


LIST_MARKER_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
LINK_DEST_RE = re.compile(r"\]\((?:[^()\\]|\\.)+\)")
AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:)[^>\n]+>")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")
INLINE_CLOSERS = frozenset("\"'’”`*_])}")


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class ListItem:
    start: int
    end: int
    level: int
    ordered: bool
    token_index: int


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    if offsets[-1] < len(text):
        offsets.append(len(text))
    return offsets


def mapped_range(offsets: Sequence[int], line_map: Sequence[int]) -> tuple[int, int]:
    start_line, end_line = (int(line_map[0]), int(line_map[1]))
    return (
        offsets[min(start_line, len(offsets) - 1)],
        offsets[min(end_line, len(offsets) - 1)],
    )


def inline_is_bold_only(inline: Any) -> bool:
    children = list(inline.children or [])
    while children and (
        children[-1].type in {"softbreak", "hardbreak"}
        or not str(children[-1].content).strip()
        and children[-1].type == "text"
    ):
        children.pop()
    while children and (
        not str(children[0].content).strip()
        and children[0].type == "text"
    ):
        children.pop(0)
    if len(children) < 3:
        return False
    if children[0].type != "strong_open" or children[-1].type != "strong_close":
        return False
    depth = 0
    for index, child in enumerate(children):
        if child.type == "strong_open":
            depth += 1
        elif child.type == "strong_close":
            depth -= 1
            if depth == 0 and index != len(children) - 1:
                return False
        elif depth <= 0 and str(child.content).strip():
            return False
    return depth == 0


def is_short_label_prefix(value: str, max_words: int = 14) -> bool:
    """Return whether a fragment is a short label that introduces following text."""
    plain = LIST_MARKER_RE.sub("", value.strip())
    plain = re.sub(r"^\s*(?:>\s*)+", "", plain)
    plain = re.sub(r"^\s*#{1,6}\s+", "", plain)
    plain = plain.replace("**", "").replace("__", "")
    plain = plain.replace("`", "").strip()
    if not plain.endswith((":", "：")):
        return False
    body = plain[:-1].strip()
    if not body or re.search(r"(?:[!?。！？]|\.\s)", body):
        return False
    return len(re.findall(r"\b[\w'-]+\b", body, flags=re.UNICODE)) <= max_words


def has_trailing_short_label(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    clauses = re.split(r"(?<=[.!?。！？])\s+", stripped)
    return is_short_label_prefix(clauses[-1])


def markdown_structure(
    md: MarkdownIt, text: str
) -> tuple[list[Any], list[Region], list[ListItem], list[tuple[int, int]]]:
    tokens = md.parse(text)
    offsets = line_offsets(text)
    top_blocks: list[Region] = []
    list_items: list[ListItem] = []
    table_rows: list[tuple[int, int]] = []
    list_stack: list[tuple[int, bool]] = []
    for index, token in enumerate(tokens):
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            list_stack.append((token.level, token.type == "ordered_list_open"))
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
        elif token.type == "list_item_open" and token.map is not None:
            start, end = mapped_range(offsets, token.map)
            ordered = list_stack[-1][1] if list_stack else False
            list_items.append(
                ListItem(start, end, token.level, ordered, index)
            )
        elif token.type == "tr_open" and token.map is not None:
            table_rows.append(mapped_range(offsets, token.map))

        if token.level != 0 or token.map is None:
            continue
        kind = ""
        if token.type == "heading_open":
            kind = "heading"
        elif token.type == "paragraph_open":
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            kind = (
                "heading"
                if inline is not None
                and inline.type == "inline"
                and inline_is_bold_only(inline)
                else "prose"
            )
        elif token.type in {"bullet_list_open", "ordered_list_open"}:
            kind = "list"
        elif token.type == "table_open":
            kind = "table"
        elif token.type in {"fence", "code_block"}:
            kind = "code"
        elif token.type == "blockquote_open":
            kind = "quote"
        elif token.type == "hr":
            kind = "hr"
        elif token.type in {"html_block"}:
            kind = "html"
        if kind:
            start, end = mapped_range(offsets, token.map)
            if kind == "prose" and is_short_label_prefix(text[start:end]):
                kind = "heading"
            top_blocks.append(Region(start, end, kind))

    top_blocks.sort(key=lambda region: (region.start, region.end))
    nonoverlap: list[Region] = []
    for region in top_blocks:
        if nonoverlap and region.start < nonoverlap[-1].end:
            continue
        nonoverlap.append(region)
    return tokens, nonoverlap, list_items, table_rows


def scan_delimited(text: str, delimiter: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    length = len(delimiter)
    while cursor < len(text):
        start = text.find(delimiter, cursor)
        if start < 0:
            break
        if start > 0 and text[start - 1] == "\\":
            cursor = start + length
            continue
        end = text.find(delimiter, start + length)
        while end >= 0 and end > 0 and text[end - 1] == "\\":
            end = text.find(delimiter, end + length)
        if end < 0:
            break
        result.append((start, end + length))
        cursor = end + length
    return result


def scan_backtick_code(text: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] != "`":
            cursor += 1
            continue
        run_end = cursor + 1
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        marker = text[cursor:run_end]
        if len(marker) >= 3 and (
            cursor == 0 or text[cursor - 1] == "\n"
        ):
            cursor = run_end
            continue
        end = text.find(marker, run_end)
        if end < 0:
            cursor = run_end
            continue
        result.append((cursor, end + len(marker)))
        cursor = end + len(marker)
    return result


def protected_regions(text: str, top_blocks: Sequence[Region]) -> list[Region]:
    regions = [
        Region(block.start, block.end, block.kind)
        for block in top_blocks
        if block.kind in {"code", "html"}
    ]
    regions.extend(Region(start, end, "inline_code") for start, end in scan_backtick_code(text))
    regions.extend(Region(start, end, "display_math") for start, end in scan_delimited(text, "$$"))
    regions.extend(Region(match.start(), match.end(), "link_destination") for match in LINK_DEST_RE.finditer(text))
    regions.extend(Region(match.start(), match.end(), "autolink") for match in AUTOLINK_RE.finditer(text))
    regions.extend(Region(match.start(), match.end(), "html_tag") for match in HTML_TAG_RE.finditer(text))
    regions.sort(key=lambda region: (region.start, -region.end))
    merged: list[Region] = []
    for region in regions:
        if not merged or region.start >= merged[-1].end:
            merged.append(region)
        elif region.end > merged[-1].end:
            previous = merged[-1]
            merged[-1] = Region(previous.start, region.end, previous.kind)
    return merged


def containing_region(position: int, regions: Sequence[Region]) -> Region | None:
    for region in regions:
        if region.start < position < region.end:
            return region
        if region.start >= position:
            break
    return None


def first_line_content(text: str, item: ListItem) -> str:
    raw = text[item.start : item.end]
    first = raw.splitlines()[0] if raw.splitlines() else raw
    return LIST_MARKER_RE.sub("", first).strip()


def strip_outer_emphasis(value: str) -> str:
    stripped = value.strip().rstrip()
    for marker in ("**", "__"):
        if stripped.startswith(marker) and stripped.endswith(marker) and len(stripped) >= 2 * len(marker):
            return stripped[len(marker) : -len(marker)].strip()
    return stripped


def is_list_label(text: str, item: ListItem) -> bool:
    content = first_line_content(text, item)
    plain = strip_outer_emphasis(content)
    if not plain:
        return False
    if plain.endswith(":"):
        return True
    if plain[-1:] in ".!?。！？":
        return False
    words = re.findall(r"\b[\w'-]+\b", plain)
    return bool(words and len(words) <= 14)


def immediate_nested_first_items(
    items: Sequence[ListItem],
) -> dict[int, int]:
    suppress: dict[int, int] = {}
    for parent in items:
        candidates = [
            child
            for child in items
            if parent.start < child.start < child.end <= parent.end
            and child.level > parent.level
        ]
        if not candidates:
            continue
        minimum_level = min(child.level for child in candidates)
        first = min(
            (child for child in candidates if child.level == minimum_level),
            key=lambda child: child.start,
        )
        suppress[parent.token_index] = first.start
    return suppress


def token_boundary_positions(
    decoded: str, offsets: Sequence[tuple[int, int]]
) -> list[int]:
    if not offsets:
        return [0]
    positions = [0]
    positions.extend(int(offsets[index][0]) for index in range(1, len(offsets)))
    positions.append(len(decoded))
    return positions


def is_word_split(text: str, position: int) -> bool:
    return (
        0 < position < len(text)
        and text[position - 1].isalnum()
        and text[position].isalnum()
    )


def snap_char_boundary(
    text: str, token_positions: Sequence[int], char_position: int
) -> int:
    return min(
        range(len(token_positions)),
        key=lambda index: (
            int(is_word_split(text, token_positions[index])),
            abs(token_positions[index] - char_position),
            index,
        ),
    )


def token_ranked_boundaries(
    text: str,
    token_positions: Sequence[int],
    char_priorities: dict[int, int],
    protected: Sequence[Region],
) -> dict[int, int]:
    ranked: dict[int, int] = {}
    for char_position, priority in char_priorities.items():
        index = snap_char_boundary(text, token_positions, char_position)
        ranked[index] = min(ranked.get(index, 9999), priority)
    for index, char_position in enumerate(token_positions[1:-1], start=1):
        region = containing_region(char_position, protected)
        if region is not None:
            priority = 1000
        elif is_word_split(text, char_position):
            priority = 100
        elif char_position > 0 and text[char_position - 1] == "\n":
            priority = 5
        else:
            priority = 20
        ranked[index] = min(ranked.get(index, 9999), priority)
    return ranked


def split_long_interval(
    start: int,
    end: int,
    ranked: dict[int, int],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[int]:
    candidates = sorted(
        {start, end}
        | {position for position in ranked if start < position < end}
    )
    best: dict[int, tuple[float, int | None]] = {start: (0.0, None)}
    for right in candidates[1:]:
        choice: tuple[float, int | None] | None = None
        for left in candidates:
            if left >= right or left not in best:
                continue
            length = right - left
            if length > max_tokens:
                continue
            short_penalty = max(0, min_tokens - length) * 50.0
            boundary_penalty = 0.0 if right == end else ranked.get(right, 1000)
            balance_penalty = abs(max_tokens * 0.75 - length) * 0.001
            candidate = (
                best[left][0]
                + short_penalty
                + boundary_penalty
                + balance_penalty,
                left,
            )
            if choice is None or candidate < choice:
                choice = candidate
        if choice is not None:
            best[right] = choice
    if end not in best:
        return list(range(start + max_tokens, end, max_tokens))
    result: list[int] = []
    cursor = end
    while cursor != start:
        parent = best[cursor][1]
        if parent is None:
            break
        if cursor != end:
            result.append(cursor)
        cursor = parent
    return sorted(result)


def coalesce_short(
    boundaries: list[int],
    *,
    hard: set[int],
    min_tokens: int,
    max_tokens: int,
) -> list[int]:
    result = sorted(set(boundaries))
    while len(result) > 2:
        short_index = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(result, result[1:])
                )
                if right - left < min_tokens
            ),
            None,
        )
        if short_index is None:
            break
        left = result[short_index]
        right = result[short_index + 1]
        remove_right = (
            right not in hard
            and short_index + 2 < len(result)
            and result[short_index + 2] - left <= max_tokens
        )
        remove_left = (
            left not in hard
            and short_index > 0
            and right - result[short_index - 1] <= max_tokens
        )
        if remove_right:
            del result[short_index + 1]
        elif remove_left:
            del result[short_index]
        else:
            break
    return result

