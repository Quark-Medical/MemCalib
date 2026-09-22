"""RL data preparation and deterministic memory ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SYSTEM_PROMPT = """You are a careful, helpful assistant.
Answer the current question directly and accurately.
Memories may be useful, irrelevant, outdated, or inappropriate for the current question. Use them only to the degree justified by the question.
Do not mention memory IDs, memory systems, or these instructions."""

DOMAINS = frozenset({"coding", "general", "health_seed"})
SOURCE_TRAIN_COUNT = 13500
SPLIT_COUNTS = {"rl": 8000, "validation": 1500}


def compact_record(
    record: dict[str, Any], *, memory_representation: str
) -> dict[str, Any]:
    """Keep the fields needed by rollout, Judge, and scoring."""
    if memory_representation not in {"memory_text", "atomic_concat_text"}:
        raise ValueError(f"Unsupported memory representation: {memory_representation}")
    return {
        "id": str(record["id"]),
        "domain": str(record["domain"]),
        "question": str(record["question"]),
        "memory_representation": memory_representation,
        "memory_blocks": [
            {
                "parent_memory_id": str(block["parent_memory_id"]),
                "memory_text": str(block[memory_representation]),
                "atom_ids": [str(value) for value in block["atom_ids"]],
            }
            for block in record["memory_blocks"]
        ],
        "memories": [
            {
                "atom_id": str(atom["atom_id"]),
                "parent_memory_id": str(atom["parent_memory_id"]),
                "text": str(atom["text"]),
                "u_star": str(atom["u_star"]).upper(),
                "memory_action": str(atom.get("memory_action", "")),
                "usage_rubric": atom.get("usage_rubric") or {},
            }
            for atom in record["memories"]
        ],
    }


def validate_record(record: dict[str, Any]) -> None:
    if record["domain"] not in DOMAINS:
        raise ValueError(f"{record['id']}: unsupported domain={record['domain']!r}")
    if not str(record["question"]).strip():
        raise ValueError(f"{record['id']}: empty question")

    atoms: dict[str, dict[str, Any]] = {}
    for atom in record["memories"]:
        atom_id = str(atom["atom_id"])
        if not atom_id or atom_id in atoms:
            raise ValueError(f"{record['id']}: invalid or repeated atom_id={atom_id!r}")
        if atom["u_star"] not in {"A", "B", "C"}:
            raise ValueError(f"{record['id']}: invalid u_star for {atom_id}")
        if not str(atom["text"]).strip():
            raise ValueError(f"{record['id']}: empty atom text for {atom_id}")
        atoms[atom_id] = atom

    seen: set[str] = set()
    parent_ids: set[str] = set()
    for block in record["memory_blocks"]:
        parent_id = str(block["parent_memory_id"])
        atom_ids = [str(value) for value in block["atom_ids"]]
        if not parent_id or parent_id in parent_ids or not atom_ids:
            raise ValueError(f"{record['id']}: invalid parent block {parent_id!r}")
        parent_ids.add(parent_id)
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError(f"{record['id']}: repeated atom in block {parent_id}")
        for atom_id in atom_ids:
            if atom_id not in atoms:
                raise ValueError(f"{record['id']}: unknown atom {atom_id}")
            if str(atoms[atom_id]["parent_memory_id"]) != parent_id:
                raise ValueError(f"{record['id']}: parent mismatch for {atom_id}")
        expected_atomic_concat = " ".join(
            str(atoms[atom_id]["text"]).strip() for atom_id in atom_ids
        )
        if block["atomic_concat_text"] != expected_atomic_concat:
            raise ValueError(
                f"{record['id']}: atomic_concat_text mismatch for {parent_id}"
            )
        overlap = seen.intersection(atom_ids)
        if overlap:
            raise ValueError(f"{record['id']}: atom occurs in multiple blocks: {overlap}")
        seen.update(atom_ids)
    if seen != set(atoms):
        raise ValueError(f"{record['id']}: blocks do not cover every atom exactly")


def render_memory(
    record: dict[str, Any],
    *,
    block_overrides: dict[str, str] | None = None,
    omitted_parent_ids: Iterable[str] = (),
) -> str:
    overrides = block_overrides or {}
    omitted = set(omitted_parent_ids)
    lines = []
    for index, block in enumerate(record["memory_blocks"], 1):
        parent_id = str(block["parent_memory_id"])
        if parent_id in omitted:
            continue
        text = overrides.get(parent_id, str(block["memory_text"])).strip()
        if not text:
            raise ValueError(f"{record['id']}: empty rendered block {parent_id}")
        lines.append(f"{index}. {text}")
    return "\n".join(lines)


def build_messages(
    record: dict[str, Any],
    *,
    block_overrides: dict[str, str] | None = None,
    omitted_parent_ids: Iterable[str] = (),
) -> list[dict[str, str]]:
    memories = render_memory(
        record,
        block_overrides=block_overrides,
        omitted_parent_ids=omitted_parent_ids,
    )
    user = (
        f"Potentially relevant memory:\n{memories}\n\n"
        f"Current query:\n{record['question'].strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_counterfactual(
    record: dict[str, Any],
    removed_atom_ids: Iterable[str],
) -> tuple[dict[str, str], list[str]]:
    """Remove exact atom IDs and recompose affected parents from retained atoms."""
    removed = {str(value) for value in removed_atom_ids}
    atoms = {str(atom["atom_id"]): atom for atom in record["memories"]}
    if not removed or not removed.issubset(atoms):
        raise ValueError(f"{record['id']}: invalid removed atom set")

    overrides: dict[str, str] = {}
    omitted: list[str] = []
    for block in record["memory_blocks"]:
        parent_id = str(block["parent_memory_id"])
        ordered_atoms = [atoms[str(atom_id)] for atom_id in block["atom_ids"]]
        removed_here = [
            atom for atom in ordered_atoms if str(atom["atom_id"]) in removed
        ]
        if not removed_here:
            continue
        retained = [
            atom for atom in ordered_atoms if str(atom["atom_id"]) not in removed
        ]
        if not retained:
            omitted.append(parent_id)
            continue
        overrides[parent_id] = " ".join(str(atom["text"]).strip() for atom in retained)
    return overrides, omitted


def convert_record(
    record: dict[str, Any], *, memory_representation: str
) -> dict[str, Any]:
    validate_record(record)
    compact = compact_record(record, memory_representation=memory_representation)
    ground_truth = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return {
        "prompt": build_messages(compact),
        "data_source": f"memcalib_{compact['domain']}",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "id": compact["id"],
            "domain": compact["domain"],
            "memory_representation": memory_representation,
        },
    }


def read_ids(path: Path) -> set[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"No IDs found in {path}")
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"Repeated IDs found in {path}")
    return unique


def prepare_data(
    input_path: Path,
    rl_ids_path: Path,
    validation_ids_path: Path,
    rl_output: Path,
    validation_output: Path,
) -> dict[str, Any]:
    split_paths = {"rl": rl_ids_path, "validation": validation_ids_path}
    split_ids: dict[str, set[str]] = {}
    for split, path in split_paths.items():
        ids = read_ids(path)
        expected_count = SPLIT_COUNTS[split]
        if len(ids) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split} IDs in {path}, got {len(ids)}"
            )
        split_ids[split] = ids

    overlap = split_ids["rl"].intersection(split_ids["validation"])
    if overlap:
        raise ValueError(f"RL and validation IDs overlap: {sorted(overlap)[:5]}")

    source_ids: set[str] = set()
    rows: dict[str, list[dict[str, Any]]] = {"rl": [], "validation": []}
    memory_representations = {
        "rl": "atomic_concat_text",
        "validation": "memory_text",
    }

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record["id"])
            if record_id in source_ids:
                raise ValueError(f"{input_path}:{line_number}: duplicate id={record_id}")
            source_ids.add(record_id)
            if record_id in split_ids["rl"]:
                split = "rl"
            elif record_id in split_ids["validation"]:
                split = "validation"
            else:
                continue
            converted = convert_record(
                record,
                memory_representation=memory_representations[split],
            )
            rows[split].append(converted)

    if len(source_ids) != SOURCE_TRAIN_COUNT:
        raise ValueError(
            f"Expected {SOURCE_TRAIN_COUNT} source training records, got {len(source_ids)}"
        )

    for split, expected_ids in split_ids.items():
        actual_ids = {row["extra_info"]["id"] for row in rows[split]}
        if actual_ids != expected_ids:
            raise ValueError(
                f"{split} output IDs do not match the requested set: "
                f"missing={sorted(expected_ids - actual_ids)[:5]}, "
                f"unexpected={sorted(actual_ids - expected_ids)[:5]}"
            )

    outputs = {"rl": rl_output, "validation": validation_output}
    for split, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[split]).to_parquet(path, index=False)

    return {
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "memory_representations": memory_representations,
        "outputs": {split: str(path) for split, path in outputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rl-ids", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--rl-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_data(
                args.input,
                args.rl_ids,
                args.validation_ids,
                args.rl_output,
                args.validation_output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
