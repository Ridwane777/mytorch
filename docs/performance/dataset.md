# Dataset analysis and train/validation/test split

Analysis for issue #12: what does the dataset contain, and how is it cut into sets that never
share a position? Related decision: [ADR-0017](../adr/0017-dataset-split.md) (Proposed).

**Status: pending measurement.** The real `dataset/` (`dataset.zip`, given by the school, not pushed)
was not available when this document was written. The scripts were run only on a small synthetic
dataset to check their logic; no figure from that run appears here, and none of the files or charts
below exists in the repository yet. Every result marked "pending measurement" is filled in by running
`sh scripts/dataset_split.sh` on the real dataset.

## Method

1. **Reading** (`scripts/dataset_split.py`): every `dataset/*/*.txt` in sorted path order, one
   position per line (`<board> <side> <castling> <ep> <halfmove> <fullmove> <label>`; the label is
   `Nothing`, `Check White|Black` or `Checkmate White|Black`). Blank lines are ignored. A line with
   fewer than 7 fields or an unknown label stops the script with `file:line` on stderr and a
   non-zero exit status.
2. **Position key**: the first four fields (board, side to move, castling, en passant). The clocks
   are ignored because the network input does not use them: the same position with other clocks is
   the same input.
3. **Statistics** (over every valid line, duplicates included): lines per label, colour (suffix of
   the label), side to move per class, distribution of the total number of pieces per class and of
   the count of each piece letter (`PNBRQKpnbrqk`) per class, number of lines, unique keys,
   duplicate lines and contradictory keys.
4. **Split** (stratified, dedupe-aware):
   - lines with the same key and the same label keep one line (the first read);
   - a key with several labels is *contradictory*: it is excluded from all sets and written to
     `contradictions.txt`;
   - for each of the five labels, the sorted keys are shuffled with `random.Random(seed)` and cut
     80 % / 10 % / 10 % (rounded to whole positions); a key goes to exactly one set; each file is
     then shuffled so the labels are mixed.
5. **Outputs**: `train.txt`, `validation.txt`, `test.txt`, `contradictions.txt` and
   `split_manifest.txt` (seed, ratios, count per set and label, SHA-256 of each file) in
   `build/dataset_split/` (ignored by Git, never committed); the CSV files below in
   `docs/performance/data/`, and the charts drawn from them by `scripts/plot_dataset.py` (Python 3
   standard library for the split; matplotlib only for the charts).

- **Reproduce**: `sh scripts/dataset_split.sh` (environment: `DATASET`, `OUT`, `SEED`; options of
  the script: `--seed`, `--train`, `--validation`, `--test`). The same dataset and seed give
  byte-identical files; compare the SHA-256 lines of `split_manifest.txt` to check two runs or two
  machines.
- **Machine and duration**: pending measurement.

## Figures stated in the issue

The distribution below is copied from issue #12; it matches the line counts of
[reading-a-fen.md](../reading-a-fen.md) but was **not re-measured by these scripts**.

| Item | Value | Source |
| --- | --- | --- |
| Total lines | 723,019 | stated in issue #12, not yet re-measured |
| `Nothing` | 300,000 | stated in issue #12, not yet re-measured |
| `Check` (White + Black) | 249,300 | stated in issue #12, not yet re-measured |
| `Checkmate` (White + Black) | 173,719 | stated in issue #12, not yet re-measured |

## Results

All of the following is **pending measurement** (run `scripts/dataset_split.sh`).

| Result | File | Value |
| --- | --- | --- |
| Lines per label and colour | `data/dataset_labels.csv` | pending measurement |
| Side to move per class; number of pieces per class and per piece type | `data/dataset_pieces.csv` | pending measurement |
| Lines, unique keys, duplicate lines, contradictory keys | `data/dataset_duplicates.csv` | pending measurement |
| Positions per set and label | `data/dataset_split.csv` | pending measurement |
| Share of lines that are duplicates | derived from the line above | pending measurement |
| Share of positions lost to contradictions | derived from the line above | pending measurement |

Charts (these images do not exist yet: they are produced by the command above):

- `data/dataset_labels.png`: lines per label, side to move per class;
- `data/dataset_pieces.png`: pieces per board by class, mean count per piece type;
- `data/dataset_split.png`: positions per label in each set.

## Limits

- Nothing above is measured on the real dataset yet; ADR-0017 stays Proposed until it is.
- Position equality is the key (board, side, castling, en passant). Two boards that are the same
  position written differently are not detected, and the same key may come from different games;
  that is intended.
- Contradictory positions are excluded, which can hide label noise from the test score; their
  number is in `dataset_duplicates.csv`.
- Small strata are cut by rounding, so the ratios are not exact; the split balances the five
  labels, not the number of pieces inside a label.
- The statistics count every line, duplicates included, so they describe the dataset as given, not
  the sets that will be trained on (those are counted in `dataset_split.csv`).
- Logic checked on a synthetic dataset only (exact and clock-only duplicates, contradictions, a
  malformed file): pairwise disjoint sets, no contradictory key in any set, about 80/10/10 per label,
  same files for the same seed, other files for another seed. This is a test of the script, not a
  measurement of the dataset.
