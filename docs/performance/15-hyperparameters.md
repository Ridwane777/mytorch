# Hyperparameter search: learning rate, batch size, hidden layers and dropout

Measurement for issue #10: are the hand-picked hyperparameters of the final network (Adam 0.001,
batches of 64, hidden layers 256 → 128, no dropout) the best of a wider space, and does a random
search find as good a combination as the full grid with a third of the trials? Related decisions:
[ADR-0028](../adr/0028-hyperparameter-search-api.md) (the search module),
[ADR-0023](../adr/0023-final-architecture.md) (the current architecture).

## Method

- **Search module**: `my_torch/Search.hpp` (ADR-0028). The chess objective,
  `scripts/hyperparam_search.cpp`, writes a configuration from `configs/final_network.conf` with the
  trial's values, builds it with `Network::fromConfig`, trains it with `Network::fit` on a
  `PositionSource` (the same path as `my_torch_analyzer --train`), and returns the **macro F1 on the
  validation split**, computed by `my_torch::report` (ADR-0026).
- **Data**: as for the trial runs ([09-trials.md](09-trials.md)): the dataset split with seed 42
  (ADR-0017), the first 10 % of the training and validation parts, the training part augmented
  (ADR-0019): **228,976 training positions, 7,155 validation positions**, prepared once for every
  trial. **The test split is never read.**
- **Space**, 3 × 3 × 2 × 2 = **36 combinations**:

  | Hyperparameter | Values | Current value |
  | --- | --- | --- |
  | Adam learning rate | 0.0003, 0.001, 0.003 | 0.001 |
  | Batch size | 32, 64, 128 | 64 |
  | Hidden layers | 256 → 128, 512 → 256 | 256 → 128 |
  | Dropout after each hidden ReLU | 0, 0.2 | 0 |

  The current configuration is combination 16 of the grid.
- **Fixed settings**: encoding `fen-attacks-901` (ADR-0016), cross-entropy with the natural class
  distribution (ADR-0018), no learning-rate schedule, default initialisation (ADR-0023), at most 30
  epochs with early stopping (patience 3, best weights restored).
- **Three steps**, run by `scripts/hyperparam_search.sh`:
  1. **Grid search**: the 36 combinations, training seed 1 (`init seed=1`, batch order `seed=42`).
  2. **Random search**: 12 combinations drawn with seed 7. It reads its scores from the grid's log,
     so it trains nothing; it answers how close a third of the budget gets to the grid's best.
  3. **Seed check**: the 3 best combinations of the grid trained again with seeds 2 and 3 (initial
     weights and batch order change, data unchanged), so that a winner by luck is not retained.
- **Timing**: wall time of each trial (`steady_clock`), written to `details.csv` with the number of
  epochs and the validation accuracy. The search runs under
  `systemd-inhibit --what=sleep:handle-lid-switch`: a first probe showed that a suspended laptop
  pauses `steady_clock` but not the elapsed time, which made a 333 s trial last 2,883 s.
- **Machine**: development laptop, Intel Core i7-1355U (10 cores, 12 threads), 14 GB RAM,
  Ubuntu 26.04.1, g++ 15.2.0 `-O2`, matrix product on every thread (ADR-0024). The laptop stayed
  usable during the search, so **times are indicative**; the scores are deterministic.
- **Reproduce**: `sh scripts/hyperparam_search.sh` (resumable: rerun it after a stop), which writes
  the CSV files and graphs to [`data/hyperparameters/`](data/hyperparameters/) through
  `scripts/plot_hyperparameters.py`.

## Results

To be written when the search ends: graphs, best combinations, grid against random search, seed
check, and the decision on the final configuration.

## Limits

- **10 % of the data**: the ranking of combinations at 10 % may differ at 100 %, larger networks
  gaining most from more data (the 512 → 256 network led the first trial series at 10 %).
- **One seed in the grid**: differences below the seed-to-seed spread measured in
  [09-trials.md](09-trials.md) are noise; the seed check covers only the 3 finalists.
- **A small space**: 4 hyperparameters, 2 or 3 values each. Weight decay, the optimizer, the
  learning-rate schedule and the number of layers stay at the values of earlier measurements.
- **Selection on the validation split**: the best of 36 scores on the same 7,155 positions is
  optimistic; only the test split, read once for the final network, gives an unbiased score.
