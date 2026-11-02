# ADR-0038: Unattended training of the CNN on one on-demand AWS instance

- **Status**: Proposed
- **Date**: 2026-11-02
- **Deciders**: to be decided by the pair

## Context
The full-data training of the CNN ([ADR-0037](0037-convolutional-layers.md)) must not depend on a
laptop that stays on. One epoch on the full augmented file (2,289,784 lines) is about 26 minutes
on the laptop: an **estimate**, the 10 % pilot of
[20-cnn.md](../performance/20-cnn.md) multiplied by 10, not a measurement, and the time on the
rented machine is unknown. With `patience=3`, a run of about a dozen epochs (the 10 % pilot ran 12)
would last about five hours; the configuration allows 30.

Constraints: the account is a managed AWS project with one Region (`eu-north-1`), no Spot or
multi-Region tricks, and the people who start the run switch their computer off afterwards. The
training must be reproducible (same script as on a laptop, `scripts/train_final.sh`), observable
from outside while it runs, and must never keep billing after it ends.

A mismatch had to be resolved: `configs/cnn_network.conf` has `epochs=30 patience=3`, while a
progress report and a snapshot of the weights every 50 epochs was wanted first, then every 5 epochs
(the default, `PROGRESS_EVERY=5`). With 30 epochs and a period of 50, no snapshot would ever be written.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **One on-demand `c7i.8xlarge` (32 vCPU) started by `scripts/aws_train.sh`, S3 for the code and the results, a role scoped to the bucket, self-termination (chosen)** | Same script and same code as locally; nothing is lost if the laptop is off; no new service beyond EC2, S3 and IAM; the instance cannot outlive the run | On-demand price; one machine, so no automatic resume if it fails |
| The same instance as Spot | Often several times cheaper | A reclaimed instance stops the training; without automatic resume, the run would simply be lost |
| SageMaker training job | Managed lifecycle, retries | A new service, a container image and an SDK to learn; the training script must be wrapped; not needed for one C++ process |
| AWS Batch | Queue, retries, Spot handling | Compute environment, job definition and image to maintain: disproportionate for one run |
| A local machine left on | Free, nothing to set up | Depends on a laptop staying on for hours; a sleep or a power cut ends the run |

## Decision
`scripts/aws_train.sh launch` uploads the code and the dataset to a private S3 bucket, creates a
role limited to that bucket (and to terminating the instances tagged for this run), and starts one
on-demand `c7i.8xlarge` in `eu-north-1` (profile `azan`) that builds and runs
`scripts/train_final.sh`. The analyzer gets `--progress-every N` (every N epochs: a line on stderr,
which the script appends to `progress.log`, and the weights in `SAVEFILE.YYYYMMDD-HHMMSS.epochN.nn`; the training
log on stdout does not change) and the run uses `--checkpoint`, written after every epoch (ADR-0032).
Every 10 minutes and at the end, the instance uploads `progress.log`, the snapshots, `checkpoint.ckpt`
and the results (never the split or augmented data), then terminates itself; if
`terminate-instances` fails it runs `shutdown -h now` (the shutdown behavior is `terminate`).
`status`, `fetch` and `cleanup` read the run back and remove the bucket and the role.

**Epoch mismatch.** `launch` computes the effective number of epochs (`EPOCHS` if set, else the
`epochs=` of the configuration) and **refuses to start, before any AWS call**, if it is below
`PROGRESS_EVERY`, telling the user to set `EPOCHS` (for instance `EPOCHS=200`) or to lower
`PROGRESS_EVERY`. `PATIENCE=0` (an override of the configuration) keeps the run from stopping before
the first snapshot. `cnn_network.conf` is left unchanged. `train_final.sh` takes the same `EPOCHS`
and rewrites only the `epochs=` key of the generated copy.

## Consequences
- ✅ The run needs no laptop; the progress, the snapshots and the checkpoint are visible in S3
  while it runs; the final `network.nn` and the checkpoint are uploaded even if the training fails.
- ✅ No new service and no change of the library: `--progress-every` only adds a callback on
  the existing epoch hook (ADR-0035), and the training output keeps its format.
- ❌ On-demand costs more than Spot (the price difference is not measured here), accepted because a
  reclaimed instance would end the run.
- ❌ No automatic resume. After a failure the user runs `fetch` (which downloads
  `checkpoint.ckpt`), `cleanup`, then launches again with `RESUME_CHECKPOINT=runs/aws/<name>/checkpoint.ckpt`
  and the same `EPOCHS`, `FRACTION` and configuration (a checkpoint from another training is refused).
- ❌ `patience=3` can stop the training before the first snapshot (for instance at epoch 12 with
  `PROGRESS_EVERY=50`): then only the final `network.nn` and the checkpoint exist. With the default
  `PROGRESS_EVERY=5` snapshots exist from epoch 5 on, with early stopping kept. A period of 50 needs
  the run to be launched with `EPOCHS=200 PATIENCE=0`
  (`train_final.sh` and `aws_train.sh` take a `PATIENCE` override, anchored like `EPOCHS`;
  `cnn_network.conf` is unchanged). With `patience=0` there is **no early stopping and no restore of
  the best weights**: `network.nn` is the last epoch, possibly overfitted, and the snapshots
  `network.nn.<date>-<time>.epochN.nn` (every 50 epochs) with `curve.png` are the safety net to pick an earlier
  epoch, at a granularity of 50 epochs (the default period is 5). A run of 200 epochs with early
  stopping off is long: a 1 % trial measured 17.8 s per epoch on a `c7i-flex.large`, about 30 min per
  full epoch and about 100 h for 200 epochs, scaled linearly and not measured on the larger
  instances. `launch` prints the effective `EPOCHS`, `PATIENCE`, `PROGRESS_EVERY` and
  `MAX_MINUTES`, and a note on the early stopping.
- ❌ If self-termination fails twice (terminate and shutdown), the instance keeps billing until
  `cleanup`: `status` must be checked after the expected duration, and the project spend limit
  (AWS Settings > Billing) is the last safeguard.
- ❌ `MAX_MINUTES` (default: effective epochs x 40 + 60) schedules `shutdown -h +MAX_MINUTES` as soon
  as the instance boots, so that a hang (snap, apt, S3, training) cannot bill forever. It can also
  cut a legitimately long run, for instance if an epoch takes more than the assumed 40 minutes:
  the final upload then does not happen, and only what the uploads every 10 minutes already sent
  (`progress.log`, snapshots, `checkpoint.ckpt`) survives. Raise `MAX_MINUTES` for a long run.
- ❌ `cleanup` keeps its state file and fails when the session has expired or when the bucket or the
  role still exists afterwards, so that a leaked resource is never forgotten; the user must then
  sign in and run it again.
- ❌ The commit recorded in `run.txt` is that of a fresh repository holding the uploaded code, not
  the commit of the laptop; the configuration checksum is the reliable link.
- Follow-up: run it once for real, move the cost in [costs.md](../business/costs.md) from
  "Estimated" to "Measured", and revisit Spot with a resume mechanism if the run is repeated.
  The script was tested without AWS: unit tests, a real small training of the same script
  (`FRACTION=0.002 EPOCHS=4 PROGRESS_EVERY=2`), and a dry run with a stub `aws`.
