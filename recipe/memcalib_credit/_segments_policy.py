"""Typed Markdown-aware policy for RL credit-assignment segmentation.

The policy is:

1. parse block structure before linguistic sentence boundaries;
2. classify inline atomic regions from their structural context;
3. attach non-propositional prefixes and trailing decorations;
4. use a constrained global fallback only when a semantic unit exceeds the
   token cap.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from recipe.memcalib_credit import _segments_base as base


@dataclass(frozen=True)
class TypedBlock:
    start: int
    end: int
    kind: str
    attach_next: bool = False


STRONG_FIELD_LINE_RE = re.compile(
    r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?"
    r"(?:"
    r"(?:\*\*|__)[^\n]{1,80}?[:：](?:\*\*|__)"
    r"|(?:\*\*|__)[^\n]{1,80}?(?:\*\*|__)[ \t]*[:：]"
    r")(?:[ \t]+|$)"
)
PLAIN_FIELD_LINE_RE = re.compile(
    r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?"
    r"(?P<label>[A-Za-z][A-Za-z0-9 _/().-]{0,60})[:：](?:[ \t]+|$)"
)
SPEAKER_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:\*\*|__)[A-Z][\w .()'-]{0,50}[:：](?:\*\*|__)"
    r"|[A-Z][\w .()'-]{0,50}[:：]"
    r")[ \t]*$"
)
STAGE_DIRECTION_RE = re.compile(
    r"^[ \t]*(?:\*\*|__)?\[[^\]\n]{1,160}\](?:\*\*|__)?[ \t]*$"
)
TECHNICAL_CONTEXT_RE = re.compile(
    r"(?i)(?:"
    r"error(?:\s+message)?|failure(?:\s+message)?|diagnostic|exception|"
    r"status(?:\s+value)?|expected(?:\s+(?:output|value|message))?|"
    r"output|returns?|reports?|raises?"
    r")[ \t]*[:=][ \t]*$"
)
TECHNICAL_LEADIN_RE = re.compile(
    r"(?i)(?:"
    r"(?:errors?|warnings?|failures?|exceptions?)(?:\s+message)?\s+"
    r"(?:such\s+as|with\s+the\s+message)"
    r"|(?:log|raise|return|report|display|output)s?.{0,40}\bmessage"
    r"|message\s+such\s+as"
    r")[ \t]*$"
)
STATUS_SIGNAL_RE = re.compile(
    r"(?i)\b(?:"
    r"mode|status|alert|synchroni[sz]ed|enabled|disabled|"
    r"metrics?|diagnostic|exception|error|warning|failed|failure|HUD"
    r")\b"
)
CODE_LITERAL_RE = re.compile(
    r"(?:"
    r"(?<![\w-])--[A-Za-z][\w-]*"
    r"|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"
    r"|\b[A-Za-z_][\w-]*\.[A-Za-z_][\w.-]*\b"
    r"|(?:^|[ \t])/(?:[\w.-]+/)+[\w.-]*"
    r"|::|->|=>|==|!=|<=|>="
    r"|[{}][^\n]{0,120}[{}]"
    r")"
)
DOTTED_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][\w-]*(?:\.[A-Za-z0-9_][\w-]*)+\b"
)
INITIALS_NAME_RE = re.compile(
    r"\b(?:[A-Z]\.){2,}[ \t]+[A-Z][A-Za-z'-]+\b"
)
SHORT_CONNECTOR_RE = re.compile(
    r"(?i)^(?:and|or|but|nor|yet|so|however|therefore|thus|then)[,;:]?$"
)
TRAILING_CLOSERS = frozenset("\"'’”`*_])}")
TERMINAL_PUNCTUATION = frozenset(".!?。！？")
COMMON_ABBREVIATIONS = frozenset(
    {
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vol",
        "vs",
    }
)
STRONG_DEPENDENCIES = frozenset(
    {
        "amod",
        "aux",
        "auxpass",
        "case",
        "compound",
        "det",
        "fixed",
        "flat",
        "mark",
        "neg",
        "nmod",
        "nummod",
        "pcomp",
        "pobj",
        "poss",
        "prep",
    }
)


def line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    next_newline = text.find("\n", end)
    line_end = len(text) if next_newline < 0 else next_newline
    return line_start, line_end


def strip_block_markup(value: str) -> str:
    result = value.strip()
    result = re.sub(r"^\s{0,3}#{1,6}\s+", "", result)
    result = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", result)
    result = result.replace("**", "").replace("__", "").strip()
    return result


def classify_block(text: str, region: Any) -> TypedBlock:
    raw = text[int(region.start) : int(region.end)]
    plain = strip_block_markup(raw)
    original_kind = str(region.kind)
    if original_kind == "heading":
        if STAGE_DIRECTION_RE.fullmatch(raw.strip()):
            return TypedBlock(region.start, region.end, "stage_direction")
        if SPEAKER_RE.fullmatch(raw.strip()):
            return TypedBlock(region.start, region.end, "speaker", True)
        return TypedBlock(region.start, region.end, "heading", True)
    if original_kind == "hr":
        return TypedBlock(region.start, region.end, "separator", True)
    if original_kind == "prose" and SHORT_CONNECTOR_RE.fullmatch(plain):
        return TypedBlock(region.start, region.end, "connector", True)
    if original_kind == "prose" and is_standalone_decoration(plain):
        return TypedBlock(region.start, region.end, "decoration")
    if original_kind == "prose" and (
        base.is_short_label_prefix(raw) or SPEAKER_RE.fullmatch(plain)
    ):
        return TypedBlock(region.start, region.end, "label", True)
    return TypedBlock(region.start, region.end, original_kind)


def typed_structure(
    md: Any, text: str
) -> tuple[list[Any], list[TypedBlock], list[Any], list[tuple[int, int]]]:
    tokens, regions, items, table_rows = base.markdown_structure(md, text)
    return (
        tokens,
        [classify_block(text, region) for region in regions],
        items,
        table_rows,
    )


def scan_quote_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    opening: int | None = None
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == '"' and not (cursor > 0 and text[cursor - 1] == "\\"):
            if opening is None:
                opening = cursor
            else:
                pairs.append((opening, cursor + 1))
                opening = None
        cursor += 1
    curly_open: int | None = None
    for cursor, char in enumerate(text):
        if char == "“":
            curly_open = cursor
        elif char == "”" and curly_open is not None:
            pairs.append((curly_open, cursor + 1))
            curly_open = None
    return sorted(set(pairs))


def _is_contextual_technical_quote(text: str, start: int, end: int) -> bool:
    content = text[start + 1 : end - 1].strip()
    if not content or "\n" in content:
        return False
    line_start, line_end = line_bounds(text, start, end)
    before = text[line_start:start].rstrip(" \t*_`")
    before_plain = before.replace("**", "").replace("__", "").replace("`", "")
    after = text[end:line_end].strip(" \t*_`")
    previous_nonblank = next(
        (
            line.strip()
            for line in reversed(text[:line_start].splitlines())
            if line.strip()
        ),
        "",
    )
    follows_speaker = bool(SPEAKER_RE.fullmatch(previous_nonblank))
    if TECHNICAL_CONTEXT_RE.search(before_plain) or TECHNICAL_LEADIN_RE.search(
        before_plain
    ):
        return True
    lead_words = re.findall(r"\b[\w'-]+\b", before_plain)
    if before_plain.endswith((":", "：")) and len(lead_words) > 6:
        return True
    if not before and not after and re.match(
        r"(?i)^(?:"
        r"(?:status|error|warning|failure|exception)[ \t]*[:=]"
        r"|(?:unable|cannot|failed|could[ \t]+not)\b"
        r")",
        content,
    ):
        return True
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if sentence.strip()
    ]
    if (
        not follows_speaker
        and not before
        and not after
        and len(sentences) >= 2
        and len(STATUS_SIGNAL_RE.findall(content)) >= 2
    ):
        return True
    words = re.findall(r"\b[\w'-]+\b", content)
    return len(words) <= 20 and bool(CODE_LITERAL_RE.search(content))


def is_contextual_technical_quote(text: str, start: int, end: int) -> bool:
    """Protect technical literals without swallowing natural quoted prose."""
    protected = _is_contextual_technical_quote(text, start, end)
    if not protected:
        return False
    content = text[start + 1 : end - 1].strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return True
    line_start, _ = line_bounds(text, start, end)
    before = text[line_start:start].rstrip(" \t*_`")
    before_plain = before.replace("**", "").replace("__", "").replace("`", "")
    technical_signal = (
        TECHNICAL_CONTEXT_RE.search(before_plain)
        or TECHNICAL_LEADIN_RE.search(before_plain)
        or STATUS_SIGNAL_RE.search(before_plain)
        or STATUS_SIGNAL_RE.search(content)
        or CODE_LITERAL_RE.search(content)
    )
    return bool(technical_signal)


def contextual_quote_regions(text: str) -> list[Any]:
    return [
        base.Region(start, end, "technical_quote")
        for start, end in scan_quote_pairs(text)
        if is_contextual_technical_quote(text, start, end)
    ]


def structural_strong_regions(text: str) -> list[Any]:
    regions: list[Any] = []
    for delimiter in ("**", "__"):
        for start, end in base.scan_delimited(text, delimiter):
            content = text[start + len(delimiter) : end - len(delimiter)].strip()
            if not content or "\n" in content or len(content) > 160:
                continue
            _, line_end = line_bounds(text, start, end)
            suffix = text[end:line_end].strip()
            label = content.endswith((":", "：")) or suffix.startswith((":", "："))
            stage = content.startswith("[") and content.endswith("]")
            short_atomic = len(re.findall(r"\b[\w'-]+\b", content)) <= 12
            if label or stage or short_atomic:
                regions.append(base.Region(start, end, "structural_strong"))
    return regions


def dotted_identifier_regions(text: str) -> list[Any]:
    return [
        base.Region(match.start(), match.end(), "dotted_identifier")
        for match in DOTTED_IDENTIFIER_RE.finditer(text)
    ] + [
        base.Region(match.start(), match.end(), "initials_name")
        for match in INITIALS_NAME_RE.finditer(text)
    ]


def protected_regions(
    text: str,
    blocks: Sequence[TypedBlock],
    table_rows: Sequence[tuple[int, int]],
) -> list[Any]:
    top_regions = [
        base.Region(block.start, block.end, block.kind)
        for block in blocks
        if block.kind in {"code", "html"}
    ]
    regions = [
        *base.protected_regions(text, top_regions),
        *contextual_quote_regions(text),
        *structural_strong_regions(text),
        *dotted_identifier_regions(text),
        *(
            base.Region(start, end, "table_row")
            for start, end in table_rows
        ),
    ]
    regions.sort(key=lambda region: (region.start, -region.end))
    merged: list[Any] = []
    for region in regions:
        if not merged or region.start >= merged[-1].end:
            merged.append(region)
        elif region.end > merged[-1].end:
            previous = merged[-1]
            merged[-1] = base.Region(previous.start, region.end, previous.kind)
    return merged


def is_emoji_component(char: str) -> bool:
    if char in {"\ufe0e", "\ufe0f", "\u200d", "\u20e3"}:
        return True
    category = unicodedata.category(char)
    return category in {"So", "Sk"} or category.startswith("M")


def is_standalone_decoration(value: str) -> bool:
    stripped = value.strip()
    if not stripped or not any(is_emoji_component(char) for char in stripped):
        return False
    return all(
        is_emoji_component(char)
        or unicodedata.category(char).startswith(("P", "S", "M"))
        for char in stripped
    )


def extend_over_trailing_emoji(text: str, position: int, limit: int) -> int:
    cursor = position
    while cursor < limit and text[cursor] in " \t":
        cursor += 1
    emoji_start = cursor
    while cursor < limit and is_emoji_component(text[cursor]):
        cursor += 1
    if cursor == emoji_start:
        return position
    while cursor < limit and (
        is_emoji_component(text[cursor])
        or text[cursor] in TERMINAL_PUNCTUATION
        or text[cursor] in TRAILING_CLOSERS
    ):
        cursor += 1
    return cursor


def visible_right(text: str, position: int, limit: int) -> tuple[str, str]:
    cursor = position
    while cursor < limit and text[cursor].isspace():
        cursor += 1
    return text[position:cursor], text[cursor:limit]


def continues_attribution_or_literal(text: str, position: int, limit: int) -> bool:
    whitespace, right = visible_right(text, position, limit)
    if not right or "\n\n" in whitespace:
        return False
    left = text[:position].rstrip()
    if right[0] in ",;:)]}":
        return True
    if right[0] in "—–" and re.match(r"^[—–][ \t]+\S", right):
        return True
    return right[0].islower() and left.endswith(tuple(TRAILING_CLOSERS))


def _normalize_sentence_end(
    text: str, position: int, limit: int
) -> int | None:
    cursor = position
    while cursor < limit and (
        text[cursor] in TRAILING_CLOSERS
        or text[cursor] in TERMINAL_PUNCTUATION
    ):
        cursor += 1
    cursor = extend_over_trailing_emoji(text, cursor, limit)
    whitespace_end = cursor
    while whitespace_end < limit and text[whitespace_end].isspace():
        whitespace_end += 1
    whitespace = text[cursor:whitespace_end]
    if "\n" in whitespace and "\n\n" not in whitespace:
        cursor = whitespace_end
    line_end = text.find("\n", cursor, limit)
    if line_end < 0:
        line_end = limit
    if re.fullmatch(r"[ \t]*\|[ \t]*", text[cursor:line_end]):
        return None
    if continues_attribution_or_literal(text, cursor, limit):
        return None
    return cursor


PARENTHETICAL_REFERENCE_RE = re.compile(
    r"^(?:[*_]{1,2})?\(\s*(?:"
    r"\[[^\]\n]+\]\([^\n)]+\)"
    r"|https?://[^\s)]+"
    r")"
)


def normalize_sentence_end(text: str, position: int, limit: int) -> int | None:
    """Keep soft-line parenthetical references attached to preceding prose."""
    normalized = _normalize_sentence_end(text, position, limit)
    if normalized is None:
        return None
    whitespace = text[position:normalized]
    if "\n" not in whitespace or "\n\n" in whitespace:
        return normalized
    right = text[normalized:limit].lstrip(" \t")
    if PARENTHETICAL_REFERENCE_RE.match(right):
        return None
    return normalized


def period_ends_abbreviation(fragment: str, index: int) -> bool:
    prefix = fragment[:index]
    match = re.search(
        r"(?iu)(?<!\w)"
        r"([^\W\d_]+(?:[’'][^\W\d_]+)?(?:\.[^\W\d_]+)*)$",
        prefix,
    )
    if match is None:
        return False
    value = match.group(1).lower()
    return len(value) == 1 or value in COMMON_ABBREVIATIONS


def punctuation_sentence_candidates(fragment: str) -> set[int]:
    candidates: set[int] = set()
    length = len(fragment)
    for index, char in enumerate(fragment):
        if char not in TERMINAL_PUNCTUATION:
            continue
        if char == "." and period_ends_abbreviation(fragment, index):
            continue
        line_start = fragment.rfind("\n", 0, index) + 1
        if char == "." and re.fullmatch(
            r"[ \t]*(?:[-+*][ \t]+)?\d+",
            fragment[line_start:index],
        ):
            continue
        cursor = index + 1
        while cursor < length and (
            fragment[cursor] in TERMINAL_PUNCTUATION
            or fragment[cursor] in TRAILING_CLOSERS
        ):
            cursor += 1
        cursor = extend_over_trailing_emoji(fragment, cursor, length)
        probe = cursor
        while probe < length and fragment[probe].isspace():
            probe += 1
        if probe <= cursor or probe >= length:
            continue
        if "\n\n" in fragment[cursor:probe]:
            candidates.add(cursor)
            continue
        next_char = fragment[probe]
        if next_char.isupper() or next_char.isdigit() or next_char in '"“#*_`':
            candidates.add(cursor)
    return candidates


def follows_ordered_list_marker(text: str, position: int) -> bool:
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position].rstrip()
    return bool(re.fullmatch(r"[ \t]*(?:[-+*][ \t]+)?\d+[.)]", prefix))


def sentence_candidates(
    nlp: Any,
    text: str,
    markdown_tokens: Sequence[Any],
    protected: Sequence[Any],
) -> set[int]:
    offsets = base.line_offsets(text)
    candidates: set[int] = set()
    for index, token in enumerate(markdown_tokens):
        if token.type != "inline" or token.map is None:
            continue
        if index > 0 and markdown_tokens[index - 1].type == "heading_open":
            continue
        start, end = base.mapped_range(offsets, token.map)
        fragment = text[start:end]
        if not fragment.strip():
            continue
        local_candidates = set(punctuation_sentence_candidates(fragment))
        doc = nlp(fragment)
        local_candidates.update(
            int(sentence.end_char)
            for sentence in doc.sents
            if 0 < int(sentence.end_char) < len(fragment)
        )
        for local_position in local_candidates:
            absolute = start + local_position
            if follows_ordered_list_marker(text, absolute):
                continue
            normalized = normalize_sentence_end(text, absolute, end)
            if normalized is None or not start < normalized < end:
                continue
            if base.containing_region(normalized, protected) is not None:
                continue
            semantic_left = text[start:normalized].rstrip().rstrip("*_`'\"”’)]}")
            raw_line_tail = text[start:normalized].rsplit("\n", 1)[-1].strip()
            if re.fullmatch(
                r"(?:[-+*]|\d+[.)])[ \t]+"
                r"(?:(?:\*\*|__).+(?:\*\*|__))",
                raw_line_tail,
            ):
                continue
            semantic_payload = re.sub(
                r"^\s*(?:>\s*)+",
                "",
                semantic_left,
            ).strip()
            semantic_tail = re.sub(
                r"^\s*(?:>\s*)+",
                "",
                semantic_payload.rsplit("\n", 1)[-1],
            ).strip()
            if is_standalone_decoration(semantic_tail):
                continue
            if not semantic_left.endswith(tuple(TERMINAL_PUNCTUATION)) and not (
                normalized > 0 and is_emoji_component(text[normalized - 1])
            ):
                continue
            if text[normalized:end].strip():
                candidates.add(normalized)
    return candidates


def repeated_field_line_starts(
    text: str,
    blocks: Sequence[TypedBlock],
) -> set[int]:
    result: set[int] = set()
    compound_blocks = [
        block
        for block in blocks
        if (
            first_line := next(
                (
                    line.strip()
                    for line in text[block.start : block.end].splitlines()
                    if line.strip()
                ),
                "",
            )
        )
        and base.is_short_label_prefix(first_line)
    ]

    def inside_compound_block(position: int) -> bool:
        return any(
            block.start < position < block.end for block in compound_blocks
        )

    cursor = 0
    lines = text.splitlines(keepends=True)
    previous_nonblank: tuple[int, str] | None = None
    previous_line: str | None = None
    compound_paragraph = False
    for line in lines:
        line_without_end = line.rstrip("\r\n")
        stripped = line_without_end.strip()
        indentation = len(line_without_end) - len(line_without_end.lstrip(" \t"))
        strong_field = bool(STRONG_FIELD_LINE_RE.match(line_without_end))
        plain_match = PLAIN_FIELD_LINE_RE.match(line_without_end)
        plain_field = bool(
            plain_match is not None
            and len(re.findall(r"\b[\w'-]+\b", plain_match.group("label"))) <= 6
        )
        if cursor > 0 and (strong_field or plain_field):
            current_is_list_item = bool(
                re.match(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+", line_without_end)
            )
            immediate_colon_intro = (
                previous_line is not None
                and previous_line.strip().endswith((":", "："))
                and current_is_list_item
            )
            nested_first_content = (
                previous_nonblank is not None
                and (
                    immediate_colon_intro
                    or (
                        indentation > previous_nonblank[0]
                        and (
                            base.is_short_label_prefix(previous_nonblank[1])
                            or bool(
                                re.fullmatch(
                                    r"(?:[-+*]|\d+[.)])[ \t]+"
                                    r"(?:(?:\*\*|__).+(?:\*\*|__))",
                                    previous_nonblank[1],
                                )
                            )
                        )
                    )
                )
            )
            if (
                not nested_first_content
                and not inside_compound_block(cursor)
                and not compound_paragraph
            ):
                result.add(cursor)
        if stripped:
            previous_nonblank = (indentation, stripped)
        previous_line = line_without_end
        if not stripped:
            compound_paragraph = False
        elif base.is_short_label_prefix(stripped):
            compound_paragraph = True
        cursor += len(line)
    return result


def adjacent_whole_quote_starts(text: str) -> set[int]:
    starts: set[int] = set()
    cursor = 0
    previous_quoted = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        quoted = (
            len(stripped) >= 2
            and stripped[0] in {'"', "“"}
            and stripped[-1] in {'"', "”"}
        )
        if previous_quoted and quoted:
            starts.add(cursor + len(line) - len(line.lstrip()))
        previous_quoted = quoted
        cursor += len(line)
    return starts


def structural_boundary_plan(
    md: Any, nlp: Any, text: str
) -> tuple[dict[int, int], set[int], list[Any], list[TypedBlock]]:
    tokens, blocks, items, table_rows = typed_structure(md, text)
    offsets = base.line_offsets(text)
    embedded_blocks = [
        TypedBlock(*base.mapped_range(offsets, token.map), "code")
        for token in tokens
        if token.map is not None
        and token.level != 0
        and token.type in {"fence", "code_block", "html_block"}
    ]
    protected = protected_regions(
        text,
        [*blocks, *embedded_blocks],
        table_rows,
    )
    priorities: dict[int, int] = {0: 0, len(text): 0}
    hard: set[int] = {0, len(text)}

    suppress_starts: set[int] = set()
    for index, block in enumerate(blocks[:-1]):
        if block.attach_next:
            suppress_starts.add(blocks[index + 1].start)
        elif (
            block.kind == "prose"
            and base.has_trailing_short_label(text[block.start : block.end])
        ):
            suppress_starts.add(blocks[index + 1].start)

    for block in blocks:
        if (
            block.start not in suppress_starts
            and block.kind != "decoration"
        ):
            priorities[block.start] = min(priorities.get(block.start, 99), 0)
        if block.kind in {"code", "stage_direction"}:
            priorities[block.end] = min(priorities.get(block.end, 99), 0)
    for block in embedded_blocks:
        priorities[block.end] = min(priorities.get(block.end, 99), 0)

    nested_first = base.immediate_nested_first_items(items)
    suppressed_items = {
        child
        for item in items
        if base.is_list_label(text, item)
        for child in [nested_first.get(item.token_index)]
        if child is not None
    }
    for item in items:
        if item.start not in suppressed_items and item.start not in suppress_starts:
            priorities[item.start] = min(priorities.get(item.start, 99), 0)
            if item.ordered:
                hard.add(item.start)

    item_starts = {item.start for item in items}
    for token in tokens:
        if (
            token.type == "paragraph_open"
            and token.level > 0
            and token.map is not None
        ):
            start, _ = base.mapped_range(offsets, token.map)
            if start not in item_starts:
                priorities[start] = min(priorities.get(start, 99), 0)
                hard.add(start)

    for row_start, _ in table_rows:
        if row_start not in suppress_starts:
            priorities[row_start] = min(priorities.get(row_start, 99), 0)
            hard.add(row_start)

    for position in repeated_field_line_starts(text, blocks):
        if (
            position not in suppress_starts
            and position not in suppressed_items
            and base.containing_region(position, protected) is None
        ):
            priorities[position] = min(priorities.get(position, 99), 0)
    for position in adjacent_whole_quote_starts(text):
        priorities[position] = min(priorities.get(position, 99), 0)
    for position in sentence_candidates(nlp, text, tokens, protected):
        priorities[position] = min(priorities.get(position, 99), 1)

    for index, char in enumerate(text):
        position = index + 1
        if base.containing_region(position, protected) is not None:
            continue
        if char in ";；":
            priorities[position] = min(priorities.get(position, 99), 2)
        elif char == "\n":
            priorities[position] = min(priorities.get(position, 99), 4)
        elif char in ":：—–":
            priorities[position] = min(priorities.get(position, 99), 7)
        elif char in ",，":
            priorities[position] = min(priorities.get(position, 99), 12)

    for block in [*blocks, *embedded_blocks]:
        if block.kind != "code":
            continue
        for position in offsets:
            if block.start < position < block.end:
                priorities[position] = min(priorities.get(position, 99), 3)
    return priorities, hard, protected, blocks


def dependency_boundary_ranks(
    nlp: Any,
    text: str,
    token_positions: Sequence[int],
    protected: Sequence[Any],
) -> dict[int, int]:
    doc_tokens = [token for token in nlp(text) if not token.is_space]
    if len(doc_tokens) < 2:
        return {}
    dense = {token.i: index for index, token in enumerate(doc_tokens)}
    crossing_diff = [0] * (len(doc_tokens) + 1)
    for token in doc_tokens:
        if token.head.i == token.i or token.head.i not in dense:
            continue
        left = min(dense[token.i], dense[token.head.i])
        right = max(dense[token.i], dense[token.head.i])
        weight = 8 if token.dep_.lower() in STRONG_DEPENDENCIES else 2
        crossing_diff[left + 1] += weight
        crossing_diff[right + 1] -= weight

    ranked: dict[int, int] = {}
    crossing = 0
    for dense_index in range(1, len(doc_tokens)):
        crossing += crossing_diff[dense_index]
        char_position = int(doc_tokens[dense_index].idx)
        if base.containing_region(char_position, protected) is not None:
            continue
        token_index = base.snap_char_boundary(text, token_positions, char_position)
        if not 0 < token_index < len(token_positions) - 1:
            continue
        ranked[token_index] = min(
            ranked.get(token_index, 9999),
            8 + crossing,
        )
    return ranked


def decoration_or_connector(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if SHORT_CONNECTOR_RE.fullmatch(stripped):
        return True
    return all(
        is_emoji_component(char)
        or unicodedata.category(char).startswith(("P", "S", "M"))
        for char in stripped
    )


def coalesce_orphans(
    boundaries: list[int],
    *,
    text: str,
    token_positions: Sequence[int],
    min_tokens: int,
    max_tokens: int,
) -> list[int]:
    result = sorted(set(boundaries))
    changed = True
    while changed and len(result) > 2:
        changed = False
        for index, (left, right) in enumerate(zip(result, result[1:])):
            fragment = text[token_positions[left] : token_positions[right]]
            if (
                is_standalone_decoration(fragment)
                and index > 0
                and right - result[index - 1] <= max_tokens
            ):
                del result[index]
                changed = True
                break
            if right - left >= min_tokens:
                continue
            if not decoration_or_connector(fragment):
                continue
            connector = bool(SHORT_CONNECTOR_RE.fullmatch(fragment.strip()))
            if (
                connector
                and index + 2 < len(result)
                and result[index + 2] - left <= max_tokens
            ):
                del result[index + 1]
                changed = True
                break
            if index > 0 and right - result[index - 1] <= max_tokens:
                del result[index]
                changed = True
                break
            if (
                index + 2 < len(result)
                and result[index + 2] - left <= max_tokens
            ):
                del result[index + 1]
                changed = True
                break
    return result


def segment_token_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    md: Any,
    nlp: Any,
    align_tokens: Callable[
        [Any, Sequence[int]], tuple[str, list[tuple[int, int]]]
    ],
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[list[int]], str, list[tuple[int, int]], list[Any]]:
    if not token_ids:
        return [], "", [], []
    decoded, offsets = align_tokens(tokenizer, token_ids)
    char_priorities, hard_chars, protected, _ = structural_boundary_plan(
        md, nlp, decoded
    )
    token_positions = base.token_boundary_positions(decoded, offsets)
    ranked = base.token_ranked_boundaries(
        decoded, token_positions, char_priorities, protected
    )
    for position, priority in dependency_boundary_ranks(
        nlp, decoded, token_positions, protected
    ).items():
        ranked[position] = min(ranked.get(position, 9999), priority)

    hard_tokens = {
        base.snap_char_boundary(decoded, token_positions, position)
        for position in hard_chars
    }
    semantic_tokens = sorted(
        {
            base.snap_char_boundary(decoded, token_positions, position)
            for position, priority in char_priorities.items()
            if priority <= 1
        }
        | {0, len(token_ids)}
    )
    boundaries = [semantic_tokens[0]]
    for start, end in zip(semantic_tokens, semantic_tokens[1:]):
        if end <= start:
            continue
        if max_tokens > 0 and end - start > max_tokens:
            boundaries.extend(
                base.split_long_interval(
                    start,
                    end,
                    ranked,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                )
            )
        boundaries.append(end)
    boundaries = base.coalesce_short(
        boundaries,
        hard=hard_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    boundaries = coalesce_orphans(
        boundaries,
        text=decoded,
        token_positions=token_positions,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    segments = [
        list(range(start, end))
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]
    covered = [index for segment in segments for index in segment]
    if covered != list(range(len(token_ids))):
        raise ValueError("Segments do not cover generated tokens exactly once")
    return segments, decoded, offsets, protected
