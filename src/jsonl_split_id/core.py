"""Deterministic train/validation/test splitting for JSON Lines datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


SPLIT_NAMES = ("train", "validation", "test")
_DEFAULT_RATIOS = (Decimal("0.8"), Decimal("0.1"), Decimal("0.1"))


def _parse_ratio(value: object, index: int) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"rasio split ke-{index} tidak valid")
    try:
        ratio = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"rasio split ke-{index} tidak valid") from None
    if not ratio.is_finite() or ratio < 0:
        raise ValueError(f"rasio split ke-{index} harus finite dan tidak negatif")
    parts = ratio.as_tuple()
    if len(parts.digits) > 12 or abs(parts.exponent) > 12:
        raise ValueError(f"rasio split ke-{index} terlalu presisi")
    return Fraction(ratio)


def _allocate(total: int, ratios: Sequence[Fraction]) -> list[int]:
    quotas = [Fraction(total) * ratio for ratio in ratios]
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remainders = [quota - count for quota, count in zip(quotas, counts)]
    leftovers = total - sum(counts)
    priority = sorted(range(len(counts)), key=lambda i: (-remainders[i], i))
    for index in priority[:leftovers]:
        counts[index] += 1
    return counts


def load_jsonl(text: str) -> list[dict[str, object]]:
    """Parse non-empty JSONL lines; errors expose line numbers but never row text."""
    if not isinstance(text, str):
        raise TypeError("input JSONL harus berupa string")

    def reject_constant(value: str) -> None:
        raise ValueError("konstanta JSON non-standar")

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"JSON tidak valid di baris {line_number}") from None
        if not isinstance(record, dict):
            raise ValueError(f"baris {line_number} harus berupa objek JSON")
        records.append(record)
    if not records:
        raise ValueError("dataset JSONL kosong")
    return records


def split_records(
    records: list[dict[str, object]],
    ratios: Sequence[object] = _DEFAULT_RATIOS,
    seed: int = 42,
    stratify_by: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Split records reproducibly; optionally apportion each scalar-label group."""
    if not isinstance(records, list) or not records:
        raise ValueError("records harus berupa list objek yang tidak kosong")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("setiap record harus berupa objek/dict")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed harus integer")
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("rasio harus berisi train, validation, dan test")
    exact_ratios = [_parse_ratio(value, index) for index, value in enumerate(ratios, start=1)]
    if sum(exact_ratios, Fraction(0)) != 1:
        raise ValueError("jumlah rasio train, validation, dan test harus tepat 1")
    if stratify_by is not None and (not isinstance(stratify_by, str) or not stratify_by.strip()):
        raise ValueError("stratify_by harus nama field non-kosong")

    splits = {name: [] for name in SPLIT_NAMES}
    if stratify_by is None:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        counts = _allocate(len(shuffled), exact_ratios)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            splits[name] = shuffled[start:start + count]
            start += count
        return splits

    groups: dict[str, list[dict[str, object]]] = {}
    for row_number, record in enumerate(records, start=1):
        if stratify_by not in record:
            raise ValueError(f"field stratifikasi tidak ada di record {row_number}")
        label = record[stratify_by]
        if isinstance(label, (dict, list)):
            raise ValueError(f"field stratifikasi harus berupa label scalar di record {row_number}")
        try:
            label_key = json.dumps(label, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError(f"field stratifikasi tidak valid di record {row_number}") from None
        groups.setdefault(label_key, []).append(record)

    for label_key, members in groups.items():
        digest = hashlib.sha256(f"{seed}\0{label_key}".encode("utf-8")).digest()
        random.Random(int.from_bytes(digest, "big")).shuffle(members)
        counts = _allocate(len(members), exact_ratios)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            splits[name].extend(members[start:start + count])
            start += count

    for index, name in enumerate(SPLIT_NAMES):
        digest = hashlib.sha256(f"{seed}\0final\0{name}".encode("utf-8")).digest()
        random.Random(int.from_bytes(digest, "big")).shuffle(splits[name])
    return splits


def write_splits(
    splits: dict[str, list[dict[str, object]]], output_dir: str | Path, force: bool = False
) -> dict[str, int]:
    """Write JSONL split files; refuse to overwrite existing outputs unless forced."""
    destination = Path(output_dir)
    targets = {name: destination / f"{name}.jsonl" for name in SPLIT_NAMES}
    if not force and any(path.exists() for path in targets.values()):
        raise FileExistsError("file output sudah ada; gunakan --force untuk menimpa")
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in targets.items():
        try:
            content = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                for record in splits[name]
            )
        except (TypeError, ValueError):
            raise ValueError("record tidak dapat diserialisasi sebagai JSON standar") from None
        path.write_text(content, encoding="utf-8")
    return {name: len(splits[name]) for name in SPLIT_NAMES}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jsonl-split-id",
        description="Bagi dataset JSONL secara reproducible ke train/validation/test.",
    )
    parser.add_argument("--output-dir", required=True, help="direktori untuk tiga file split")
    parser.add_argument("--train", type=Decimal, default=Decimal("0.8"), help="rasio train (default 0.8)")
    parser.add_argument("--validation", type=Decimal, default=Decimal("0.1"), help="rasio validation (default 0.1)")
    parser.add_argument("--test", type=Decimal, default=Decimal("0.1"), help="rasio test (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="seed deterministik")
    parser.add_argument("--stratify-by", help="nama field label untuk split terstratifikasi")
    parser.add_argument("--force", action="store_true", help="izinkan menimpa file output yang sudah ada")
    args = parser.parse_args(argv)
    try:
        records = load_jsonl(sys.stdin.read())
        splits = split_records(
            records,
            ratios=(args.train, args.validation, args.test),
            seed=args.seed,
            stratify_by=args.stratify_by,
        )
        sizes = write_splits(splits, args.output_dir, force=args.force)
    except (OSError, TypeError, ValueError) as error:
        print(f"jsonl-split-id: {error}.", file=sys.stderr)
        return 2
    print(json.dumps({"total": len(records), "splits": sizes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
