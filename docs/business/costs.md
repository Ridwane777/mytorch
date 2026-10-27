# Costs

Two tables that are never mixed: what we **measured**, and what we **estimated**, each estimate with
its method so that anyone can redo it with their own numbers.

## 1. Measured

| Item | Value | Source |
| --- | --- | --- |
| Training of the pushed network, whole augmented dataset (2,289,784 lines), 14 epochs | 6,285 s (1 h 45) on a laptop, i7-1355U, 12 threads; peak memory 430 MB | [10-final-network.md](../performance/10-final-network.md), `run.txt` |
| Same network without augmentation (572,446 lines), 9 epochs | 4,109 s on a shared 8-core VPS (time inflated by the other users); peak 221 MB | same |
| Architecture trials: 8 networks on 10 % of the dataset, 8 at a time | 5,260 s on the VPS | [09-trials.md](../performance/09-trials.md), `data/trials_run.txt` |
| Prediction of 71,555 positions with the pushed network | 2.8 s on the laptop, peak 148 MB, about 39 µs per position | `10-final-network.md` |
| Network file shipped | 5.4 MB (text, plain git) | `my_torch_network_final.nn` |
| Effect of the threaded matrix product on training | 2.4 times faster from 1 to 12 threads | [12-matmul.md](../performance/12-matmul.md) |
| Project activity | 121 commits, 32 merged pull requests, from 2026-10-21 to 2026-10-27 | `git log`, GitHub |
| Code size | about 3,700 lines for the library, 1,200 for the chess part, 19 test files | `wc -l` |
| Software paid for | none: every dependency is free ([licenses.md](../legal/licenses.md)) | |
| GPU used | none: the code runs on the CPU only; the rented A5000 was not used for any measurement | |

## 2. Estimated

| Item | Estimate | Method |
| --- | --- | --- |
| Energy of the final training | between 0.03 and 0.10 kWh | Processor power between its base power (15 W) and its maximum turbo power (55 W), Intel figures for the i7-1355U, times 1.75 h. The wall power was not measured; screen and rest of the laptop not counted |
| Electricity cost of the final training | a few cents of euro | The energy above times the user's tariff in €/kWh. With 0.25 €/kWh (an assumed tariff, to replace by the real one): 0.01 to 0.03 € |
| Running cost for a user | none | The programs run on the user's machine: no server, no API, no subscription |
| Developer time | not tracked | Hours were not logged. To estimate it: hours per member times an hourly rate; the measured activity above (7 days, 121 commits) is the only reliable proxy |
| VPS and GPU rental | not written | Their prices were not recorded in the project; a rental would only pay off for networks much larger than ours (see below) |

## 3. Reading
- The cost of the project is almost entirely **people's time**: the compute fits on a laptop, in
  less than two hours for the final network.
- Making training cheaper in time is already done for the products (ADR-0024); the next steps and
  their limits are in [12-matmul.md](../performance/12-matmul.md). A GPU would only become worth its
  rental for networks an order of magnitude larger (ADR-0024, follow-up).
- For the teaching product of the [business plan](business-plan.md), the important number is the
  first line of the measured table: a student can train the full example on an ordinary laptop.
