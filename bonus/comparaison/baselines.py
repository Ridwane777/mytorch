"""Classical baselines on the chess positions (#43, ADR-0043).

One model family per call: logistic regression, random forest, scikit-learn's MLPClassifier or
LightGBM, trained on the same bytes as our network (`pytorch_reference export`, encoding
fen-attacks-901, split seed 42). Every setting of the family's small grid is trained on the training
part and scored on the validation part; the best macro F1 is kept, and only that model is scored on
the test part, once. The test part never chooses anything.

Usage: python baselines.py --model logreg|forest|mlp|lightgbm --train PREFIX --valid PREFIX
           --test PREFIX --width 901 --threads T --out-dir OUT
Writes OUT/grid.csv (one line per setting: validation scores and training seconds), OUT/scores.txt
(the test scores of the chosen setting, in the format of scripts/score.cpp), OUT/confusion.csv,
OUT/summary.csv (one line: model, setting, test accuracy and macro F1, training and prediction
seconds, size of the pickled model in bytes) and OUT/model.pkl.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


def read_bytes(prefix: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    """As mytorch_net.read_bytes, without importing PyTorch (not needed for these baselines)."""
    x = np.fromfile(f"{prefix}.x.u8", dtype=np.uint8)
    y = np.fromfile(f"{prefix}.y.u8", dtype=np.uint8)
    return x.reshape(-1, width), y


def grid(model: str, threads: int, seed: int):
    """(name, estimator) for every setting of the family. Kept small: a few hours in all."""
    if model == "logreg":
        return [(f"C={c}", LogisticRegression(C=c, max_iter=300, tol=1e-4, random_state=seed))
                for c in (0.1, 1.0, 10.0)]
    if model == "forest":
        return [(f"trees={n} min_leaf={leaf}",
                 RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, max_features="sqrt",
                                        n_jobs=threads, random_state=seed))
                # Fully grown trees on 572,446 rows: 100 trees took 7.4 GB, and 300 next to the best
                # one so far were killed by the 15 GB machine. The grid varies the leaf size only.
                for n, leaf in ((100, 1), (100, 5), (100, 20))]
    if model == "mlp":
        # Our final architecture (901 -> 512 -> 256 -> 5, ReLU, Adam 0.001, batch 384), fixed
        # 15 epochs, no early stopping: what a standard library gives with the same shape.
        return [(f"hidden=512,256 lr={lr} epochs=15",
                 MLPClassifier(hidden_layer_sizes=(512, 256), learning_rate_init=lr,
                               batch_size=384, max_iter=15, shuffle=True, random_state=seed))
                for lr in (0.001, 0.0003)]
    if model == "lightgbm":
        import lightgbm

        return [(f"leaves={leaves} lr=0.1 rounds<=2000",
                 lightgbm.LGBMClassifier(num_leaves=leaves, learning_rate=0.1, n_estimators=2000,
                                         n_jobs=threads, random_state=seed, verbose=-1))
                for leaves in (63, 255)]
    raise SystemExit(f"baselines.py: unknown model {model}")


def fit(model: str, estimator, x, y, x_valid, y_valid):
    if model == "lightgbm":
        import lightgbm

        # Early stopping on the validation part: it chooses the number of rounds, like patience
        # chooses our epochs; the test part is still never seen.
        estimator.fit(x, y, eval_set=[(x_valid, y_valid)], eval_metric="multi_logloss",
                      callbacks=[lightgbm.early_stopping(20, verbose=False)])
    else:
        estimator.fit(x, y)


def scores_text(y, predicted) -> str:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=range(len(LABELS)), zero_division=0)
    lines = [f"positions scored: {len(y)}",
             f"accuracy: {np.mean(y == predicted):.4f}",
             f"macro F1: {f1_score(y, predicted, average='macro'):.4f}"]
    for k, label in enumerate(LABELS):
        lines.append(f"{label}: precision {precision[k]:.4f}, recall {recall[k]:.4f}, "
                     f"F1 {f1[k]:.4f}, support {support[k]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("--model", "--train", "--valid", "--test", "--out-dir"):
        parser.add_argument(name, required=True)
    parser.add_argument("--width", type=int, default=901)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    threads = args.threads or -1

    # float32: half the memory of float64 (572,446 x 901 is 2.1 GB), and every family accepts it.
    x, y = read_bytes(args.train, args.width)
    x_valid, y_valid = read_bytes(args.valid, args.width)
    x, x_valid = x.astype(np.float32), x_valid.astype(np.float32)

    best = None
    with open(out / "grid.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "setting", "valid_accuracy", "valid_macro_f1", "train_seconds"])
        for name, estimator in grid(args.model, threads, args.seed):
            start = time.perf_counter()
            fit(args.model, estimator, x, y, x_valid, y_valid)
            seconds = time.perf_counter() - start
            predicted = estimator.predict(x_valid)
            accuracy = float(np.mean(predicted == y_valid))
            macro = float(f1_score(y_valid, predicted, average="macro"))
            writer.writerow([args.model, name, f"{accuracy:.4f}", f"{macro:.4f}", f"{seconds:.1f}"])
            f.flush()
            print(f"{args.model} {name}: validation accuracy {accuracy:.4f}, macro F1 {macro:.4f}, "
                  f"{seconds:.0f} s", flush=True)
            if best is None or macro > best[2]:
                best = (name, None if args.model == "forest" else estimator, macro, seconds)
            del estimator

    name, estimator, _, train_seconds = best
    if estimator is None:
        # A grown forest takes several GB: keeping the best one while the next trains exceeded the
        # memory of the machine, so each forest is freed after validation and the chosen setting
        # is trained again (same seed, same trees) for the test.
        estimator = dict(grid(args.model, threads, args.seed))[name]
        start = time.perf_counter()
        fit(args.model, estimator, x, y, x_valid, y_valid)
        train_seconds = time.perf_counter() - start
    del x, y
    x_test, y_test = read_bytes(args.test, args.width)
    x_test = x_test.astype(np.float32)
    start = time.perf_counter()
    predicted = estimator.predict(x_test)
    predict_seconds = time.perf_counter() - start

    (out / "scores.txt").write_text(scores_text(y_test, predicted))
    np.savetxt(out / "confusion.csv", confusion_matrix(y_test, predicted, labels=range(len(LABELS))),
               fmt="%d", delimiter=",")
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = (out / "model.pkl").stat().st_size
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "setting", "test_accuracy", "test_macro_f1", "train_seconds",
                         "predict_seconds", "model_bytes"])
        writer.writerow([args.model, name, f"{np.mean(predicted == y_test):.4f}",
                         f"{f1_score(y_test, predicted, average='macro'):.4f}",
                         f"{train_seconds:.1f}", f"{predict_seconds:.2f}", size])
    print(f"{args.model}: chose {name}; test scores in {out / 'scores.txt'}", flush=True)


if __name__ == "__main__":
    main()
