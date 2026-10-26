#!/usr/bin/env python3
"""Handles class imbalance on a training file for scripts/bench_imbalance.sh (ADR-0018).

Usage: resample_imbalance.py MODE SEED TRAIN_FILE [CONFIG]

TRAIN_FILE holds one "FEN label" position per line. The class counts before and after go to
stderr, the result to stdout:
  none     the file unchanged (shuffled with the seed, like the other modes);
  over     every class below the majority count is completed with positions drawn with replacement
           from itself, up to the majority count; nothing is removed;
  under    every class is drawn without replacement down to the minority count;
  weights  prints the loss weights "w1,w2,..." (inverse frequency, normalised to mean 1) in the
           order of the labels of CONFIG (a configuration file, docs/configuration.md).
All draws use random.Random(SEED), over classes taken in alphabetical order: the same input and
seed always give the same output. The test set is never given to this script.
"""
import random
import sys
from collections import defaultdict


def label_of(line):
    return " ".join(line.split()[6:])


def config_labels(path):
    """The labels in the order of the `labels N` block of a configuration file."""
    with open(path) as f:
        lines = [line.strip() for line in f]
    for i, line in enumerate(lines):
        if line.startswith("labels "):
            count = int(line.split()[1])
            return lines[i + 1:i + 1 + count]
    raise SystemExit(f"resample_imbalance.py: no 'labels' block in {path}")


def counts_text(title, groups):
    return title + ": " + ", ".join(f"{k} {len(v)}" for k, v in sorted(groups.items()))


def weights(groups, labels):
    for label in labels:
        if not groups.get(label):
            raise SystemExit(f"resample_imbalance.py: no training position labelled '{label}': "
                             "cannot compute inverse frequency weights")
    unknown = sorted(set(groups) - set(labels))
    if unknown:
        raise SystemExit(f"resample_imbalance.py: labels {unknown} are not in the configuration")
    inverse = [1.0 / len(groups[label]) for label in labels]
    mean = sum(inverse) / len(inverse)
    return [w / mean for w in inverse]


def main(mode, seed, train, config=None):
    if mode not in ("none", "over", "under", "weights"):
        raise SystemExit(f"resample_imbalance.py: unknown mode '{mode}'")
    rng = random.Random(int(seed))
    with open(train) as f:
        lines = [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]
    groups = defaultdict(list)
    for line in lines:
        groups[label_of(line)].append(line)
    print(counts_text("input counts", groups), file=sys.stderr)

    if mode == "weights":
        if config is None:
            raise SystemExit("resample_imbalance.py: the weights mode needs a configuration file")
        print(",".join(f"{w:.6g}" for w in weights(groups, config_labels(config))))
        return

    out = []
    for label in sorted(groups):
        group = groups[label]
        if mode == "over":
            target = max(len(g) for g in groups.values())
            out.extend(group + rng.choices(group, k=target - len(group)))
        elif mode == "under":
            target = min(len(g) for g in groups.values())
            out.extend(rng.sample(group, target))
        else:
            out.extend(group)
    rng.shuffle(out)
    resampled = defaultdict(list)
    for line in out:
        resampled[label_of(line)].append(line)
    print(counts_text(f"{mode} counts", resampled), file=sys.stderr)
    sys.stdout.writelines(out)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
