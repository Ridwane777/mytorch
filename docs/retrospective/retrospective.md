# Retrospective: what we could have done better

Written on 2026-10-27, after the final network ([10-final-network.md](../performance/10-final-network.md)).
Each point names an alternative, the threshold where it becomes the better choice, and its own cost.
The facts come from the ADRs, the measurements and `git log`.

## 1. In one sentence
The library and the network work and every choice is measured, but we built the end-to-end path and
the profiling late, so several decisions were made on samples and slow code, and had to be checked
again at full scale.

## 2. What would have reduced the time

| What happened | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| The full pipeline (split → augment → train → score) came on day 6 (`scripts/train_final.sh`, ADR-0022). Training on the whole dataset then showed that encoding the whole file at once would need about 10.4 GB: we had to rework the data path (ADR-0021) | A **walking skeleton**: the thinnest end-to-end run on the full dataset on day 1, even with a one-layer network, then improve each stage | As soon as the data is larger than a quick test (here, from the first day: 723,019 positions) | Half a day at the start, and a first network with poor scores |
| The matrix product stayed a plain loop until the end (ADR-0003 already listed the optimisation as a task). The 8 architecture trials took 5,260 s with one thread per run; since ADR-0024 the products alone are 4 to 5 times faster on one thread (`12-matmul.md`), so the trials would have been shorter, by an amount we did not measure | **Profile on day 2** (`perf`, ten minutes) and do the blocked product before the experiments | Whenever experiments are repeated: every gain is multiplied by the number of runs | One day of work, done earlier |
| An early estimate of 17 minutes per epoch was wrong: the measurement gave 3 minutes (`11-training-memory.md`) | **Measure before planning**, never plan on an estimate | Always | A short run |
| Encodings (ADR-0016) and class imbalance (ADR-0018) were compared on 20,000 positions and one network size, because the code was slow and memory-bound at that time | Redo the decisive comparisons at full scale once training is fast | When the full-scale run fits in an hour, which is the case since ADR-0021 and ADR-0024 | About one hour of compute per comparison |

## 3. What would have reduced the costs or the risks

| What happened | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Part of the measurements ran on a shared VPS: results are bit-identical, but its times are unusable (other users, load around 6 on 8 cores) | Measure times on **one dedicated machine** only, and use shared machines for results, not for times | Every time a time is published | A slower turnaround on one laptop |
| A rented GPU (A5000) was available but useless: the code runs on the CPU in double precision | Keep the CPU, as we did; a GPU path (CUDA) only pays for networks about ten times larger (ADR-0024) | Above a few million parameters | A second code path to write and test, a CUDA dependency |
| Double precision everywhere (ADR-0003): twice the memory and time of `float` | `float` for the weights and products, with the finite-difference tests kept in double | When training time is the bottleneck and the bit-identical results across versions are no longer needed | Changes every saved network and every reference result |
| Text network file, 2.7 times larger than binary, without Adam's state (ADR-0012) | A binary format with the optimizer state | At about 5 million parameters (GitHub's limits) or to resume a training exactly | A new format version and its tests |
| Work was reassigned late (four issues moved between members on the last day) | Re-balance the issues at a short weekly check of the board | From the first week | Fifteen minutes per week |

## 4. What would have made the product better
- **Check against checkmate** is where the network fails: 91 % of its 2,754 errors on the test split.
  The input (attacked squares, ADR-0016) does not say whether the king has a legal move. Alternative:
  add a feature for the number of legal king moves, or for every legal move of the side in check.
  Better as soon as the remaining errors are mostly of this kind, which they are. Cost: it moves chess
  rules into the encoding (less general), and a new encoding means retraining.
- **One seed per run**: only the trials measured the seed spread (±0.3 to ±0.6 points of macro F1,
  [09-trials.md](../performance/09-trials.md)). Running three seeds of every decisive comparison costs three
  times the compute, and becomes affordable now that training is faster.
- **Automatic hyperparameter search** (#10) was not done: learning rate, batch size and layer sizes
  come from the trials and the defaults. A small random search on 10 % of the data would cost a few
  hours of compute.

## 5. What we would keep without hesitation
- **ADRs written when the decision is made**, with their ❌: they made every choice defendable and
  every later change visible (ADR-0001 abandoned, superseded by ADR-0002).
- **A finite-difference test for every gradient**, sanitizers (`make debug`), ThreadSanitizer for the
  thread pool.
- **Bit-identical results** whatever the machine and the number of threads: the VPS and the laptop
  could share the work, and every optimisation is checked by comparing files.
- **A script and a graph for every number** (`scripts/`, `docs/performance/`), and the test split
  never used to choose.
- **The contract between the two work packages** (`include/my_torch/`), changed only by reviewed pull
  requests.

## 6. Next actions
1. Decide ADR-0024 and ADR-0025 as a pair (both "Proposed").
2. Check the tool versions on the autograder image (`docs/dependencies.md`).
3. Try the legal-king-moves feature against the check/checkmate confusion (section 4).
4. Hyperparameter search (#10) and the remaining benchmarks (#18), now that training is fast.
5. The bonuses: ensembles, normalisation, convolution, comparison with PyTorch (#26 to #44).
