#!/usr/bin/env python3
"""Dataset statistics and a dedupe-aware, stratified train/validation/test split (issue #12).

Reads <dataset>/*/*.txt (files in sorted path order), one position per line:
    <board> <side> <castling> <ep> <halfmove> <fullmove> <label...>
The label is Nothing, Check White|Black or Checkmate White|Black. Blank lines are ignored.

Position key = fields 1 to 4 (board, side, castling, en passant); the clocks are ignored, so the
same position seen at two move numbers is the same key.
  - Several lines with the same key and the same label: one line is kept (the first read).
  - Several labels for one key (contradiction): the key is excluded from every split and listed in
    contradictions.txt.
  - A key goes to exactly one split, so no position leaks from train to validation or test.
  - The split is stratified by the full 5-way label: within each label, the sorted keys are shuffled
    with random.Random(seed) and cut by the ratios. The same dataset and seed give the same files.

Writes to --out: train.txt validation.txt test.txt contradictions.txt split_manifest.txt
Writes to --stats: dataset_labels.csv dataset_pieces.csv dataset_duplicates.csv dataset_split.csv

Only the Python standard library is used. Run it as `python3 -I scripts/dataset_split.py` (see
scripts/dataset_split.sh).
"""
import argparse
import csv
import hashlib
import math
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
CLASSES = ["Nothing", "Check", "Checkmate"]
PIECES = "PNBRQKpnbrqk"
SPLITS = ["train", "validation", "test"]


class DatasetError(Exception):
    """A problem with the dataset or the options, reported on stderr without a traceback."""


def colour_of(label):
    parts = label.split()
    return parts[1] if len(parts) > 1 else "none"


def class_of(label):
    return label.split()[0]


def read_dataset(dataset):
    """Returns (stats, groups). groups maps a key to {label: first line read with this label}."""
    root = Path(dataset)
    files = sorted(str(p) for p in root.glob("*/*.txt")) if root.is_dir() else []
    if not files:
        raise DatasetError(
            f"{dataset}/ not found or empty: extract the dataset (unzip dataset.zip, provided by Epitech) at "
            "the root of the repository, or pass its directory with --dataset")

    label_counts = Counter()
    piece_hist = {c: {m: Counter() for m in ["total", *PIECES, "side"]} for c in CLASSES}
    groups = {}
    lines = 0
    for path in files:
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for number, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    fields = line.split()
                    if len(fields) < 7:
                        raise DatasetError(
                            f"{path}:{number}: malformed line, expected at least 7 fields "
                            f"(board side castling ep halfmove fullmove label), got {len(fields)}")
                    label = " ".join(fields[6:])
                    if label not in LABELS:
                        raise DatasetError(f"{path}:{number}: unknown label '{label}'")
                    lines += 1
                    label_counts[label] += 1
                    board = fields[0]
                    hist = piece_hist[class_of(label)]
                    counts = {p: board.count(p) for p in PIECES}
                    hist["total"][sum(counts.values())] += 1
                    for p in PIECES:
                        hist[p][counts[p]] += 1
                    hist["side"][fields[1]] += 1
                    key = " ".join(fields[:4])
                    by_label = groups.setdefault(key, {})
                    by_label.setdefault(label, " ".join(fields))
        except (OSError, UnicodeDecodeError) as e:
            raise DatasetError(f"{path}: cannot be read as text: {e}") from e
    if lines == 0:
        raise DatasetError(f"{dataset}/ holds no position (only blank lines)")
    return {"lines": lines, "labels": label_counts, "pieces": piece_hist}, groups


def cut(n, train, validation):
    """Boundaries of the three slices of n items: [0:a] train, [a:b] validation, [b:n] test."""
    a = min(n, math.floor(n * train + 0.5))
    b = min(n, max(a, math.floor(n * (train + validation) + 0.5)))
    return a, b


def split_keys(groups, seed, train, validation):
    """Returns (assignment, contradictions): split name -> [(key, label)], and the excluded keys."""
    contradictions = sorted(k for k, by_label in groups.items() if len(by_label) > 1)
    by_stratum = {label: [] for label in LABELS}
    for key, by_label in groups.items():
        if len(by_label) == 1:
            (label,) = by_label
            by_stratum[label].append(key)

    rng = random.Random(seed)
    assignment = {name: [] for name in SPLITS}
    for label in LABELS:
        keys = sorted(by_stratum[label])
        rng.shuffle(keys)
        a, b = cut(len(keys), train, validation)
        assignment["train"] += [(k, label) for k in keys[:a]]
        assignment["validation"] += [(k, label) for k in keys[a:b]]
        assignment["test"] += [(k, label) for k in keys[b:]]
    for name in SPLITS:  # mix the labels inside each file
        rng.shuffle(assignment[name])
    return assignment, contradictions


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset statistics and a dedupe-aware stratified train/validation/test split.")
    parser.add_argument("--dataset", default="dataset", help="dataset directory (default dataset)")
    parser.add_argument("--out", default="build/dataset_split",
                        help="directory for the split files (default build/dataset_split)")
    parser.add_argument("--stats", default="docs/performance/data",
                        help="directory for the statistics CSV files (default docs/performance/data)")
    parser.add_argument("--seed", type=int, default=42, help="shuffling seed (default 42)")
    parser.add_argument("--train", type=float, default=0.8, help="training share (default 0.8)")
    parser.add_argument("--validation", type=float, default=0.1,
                        help="validation share (default 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="test share (default 0.1)")
    args = parser.parse_args()

    try:
        ratios = [args.train, args.validation, args.test]
        if min(ratios) < 0 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
            raise DatasetError(
                f"--train, --validation and --test must be >= 0 and sum to 1 (got {sum(ratios)})")

        stats, groups = read_dataset(args.dataset)
        assignment, contradictions = split_keys(groups, args.seed, args.train, args.validation)

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stats_dir = Path(args.stats)
        stats_dir.mkdir(parents=True, exist_ok=True)

        split_counts = {name: Counter() for name in SPLITS}
        for name in SPLITS:
            lines = []
            for key, label in assignment[name]:
                lines.append(groups[key][label])
                split_counts[name][label] += 1
            write_lines(out / f"{name}.txt", lines)
        contradiction_lines = [groups[k][label] for k in contradictions for label in sorted(groups[k])]
        write_lines(out / "contradictions.txt", contradiction_lines)

        manifest = [
            f"seed {args.seed}",
            f"ratios train {args.train} validation {args.validation} test {args.test}",
            "position key: fields 1 to 4 (board side castling en-passant); clocks ignored",
            f"dataset lines {stats['lines']} unique keys {len(groups)} "
            f"contradictory keys {len(contradictions)}",
        ]
        for name in SPLITS:
            manifest.append(f"{name} total {sum(split_counts[name].values())}")
            manifest += [f"{name} {label} {split_counts[name][label]}" for label in LABELS]
        for name in [*SPLITS, "contradictions"]:
            manifest.append(f"sha256 {name}.txt {sha256(out / (name + '.txt'))}")
        write_lines(out / "split_manifest.txt", manifest)

        # Statistics (over all valid lines of the dataset, duplicates included).
        write_csv(stats_dir / "dataset_labels.csv", ["label", "colour", "count"],
                  [[label, colour_of(label), stats["labels"][label]] for label in LABELS])
        rows = []
        for cls in CLASSES:
            for metric, counter in stats["pieces"][cls].items():
                rows += [[cls, metric, value, counter[value]]
                         for value in sorted(counter, key=str if metric == "side" else int)]
        write_csv(stats_dir / "dataset_pieces.csv", ["class", "metric", "value", "count"], rows)
        distinct_pairs = sum(len(by_label) for by_label in groups.values())
        write_csv(stats_dir / "dataset_duplicates.csv", ["metric", "value"], [
            ["lines", stats["lines"]],
            ["unique_keys", len(groups)],
            ["duplicate_lines", stats["lines"] - distinct_pairs],
            ["contradictory_keys", len(contradictions)],
        ])
        write_csv(stats_dir / "dataset_split.csv", ["split", "label", "count"],
                  [[name, label, split_counts[name][label]] for name in SPLITS for label in LABELS])
    except DatasetError as e:
        print(f"dataset_split.py: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"dataset_split.py: {e}", file=sys.stderr)
        return 1

    print(f"{stats['lines']} lines, {len(groups)} unique keys, "
          f"{stats['lines'] - distinct_pairs} duplicate lines, "
          f"{len(contradictions)} contradictory keys (excluded)")
    for name in SPLITS:
        print(f"{name}: {sum(split_counts[name].values())} positions")
    print(f"split written to {out}, statistics to {stats_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
