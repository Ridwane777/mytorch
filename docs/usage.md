# Usage: from a configuration to predictions

```bash
make                                              # builds my_torch_analyzer and my_torch_generator

# 1. Write blank networks from a configuration (docs/configuration.md)
./my_torch_generator configs/basic_network.conf 1          # writes basic_network_1.nn

# 2. Train: each line of the file is a FEN followed by its expected output
./my_torch_analyzer --train --save trained.nn basic_network_1.nn train.txt
./my_torch_analyzer --train basic_network_1.nn train.txt   # without --save: overwrites the file

# 2b. A long training: save its state after every epoch; after a stop, rerun the same command
./my_torch_analyzer --train --save trained.nn --checkpoint run.ckpt basic_network_1.nn train.txt

# 3. Predict: one answer per position, in file order (expected outputs, if present, are ignored)
./my_torch_analyzer --predict trained.nn chessboards.txt
```

## What each mode does

| Mode | Reads | Writes |
| --- | --- | --- |
| `--train` | The network file (layers, weights, input encoding, labels, training settings) and a file of FENs with expected outputs | One line per epoch on standard output (loss and accuracy, and validation figures when `validation` > 0), then the trained network to SAVEFILE, or LOADFILE without `--save` |
| `--predict` | The network file and a file of FENs (expected outputs optional) | Only the predictions on standard output, one per line |

- The answer printed is the label of the most probable output. A network whose labels are
  `Nothing`, `Check` and `Checkmate` answers without colors; one with five labels answers with them.
- In `--train`, an expected output trains the label equal to it, or else the label equal to its
  first word (`Check White` trains `Check`).
- With `--checkpoint FILE` (ADR-0032, an option beyond the subject's usage), the whole training
  state (weights, optimizer memory, generators, history, early stopping) is written to FILE after
  every epoch. If FILE exists, training resumes after its last epoch and ends with the same weights
  as an uninterrupted run; an epoch cut in the middle is run again. A checkpoint from another
  training (other settings, layers or data size) is refused: delete it to start again. Raising
  `epochs` in the network file and rerunning extends a finished training.
- With `--progress-every N` (an option beyond the subject's usage, `--train` only, N a positive
  integer, ADR-0038), every N epochs the epoch's measurements are printed on standard error and the
  weights of that epoch are saved into `SAVEFILE.YYYYMMDD-HHMMSS.epochN.nn` (the time of the save, so that no snapshot replaces another; or `LOADFILE.<time>.epochN.nn` without
  `--save`). Standard output is unchanged.
- Training settings come from the network file: loss, optimizer, scheduler, epochs, batch size,
  seed, early stopping and the `validation` fraction held out from the training file.
- Every input is checked before anything is printed or written. On an error the message goes to
  standard error and the exit code is 84.

Design choices: [ADR-0014](adr/0014-analyzer-command-line.md).

## Reproducing the final network

```bash
sh scripts/train_final.sh configs/final_network.conf            # whole dataset, writes runs/final_network/
FRACTION=0.01 sh scripts/train_final.sh configs/final_network.conf /tmp/try   # quick check on 1 %
```

The script splits the dataset, augments the training part only, appends the untouched validation
part at the end of the training file (`validation_tail`), generates and trains the network, then
scores it on the test part with the library's metrics (`scripts/score.cpp`, ADR-0026): `scores.txt` (precision, recall and F1 per class), `confusion.png`,
`curve.png` and `run.txt` (commit, configuration checksum, machine, times, memory). Running it twice
gives the same network byte for byte. `EVAL=validation` scores the validation split instead of the
test split (for trial runs, so that choices never look at the test split); `BUILD=0` skips the build
for parallel runs. `sh scripts/trial_runs.sh` trains every configuration of `configs/trials/` in
parallel and gathers their validation scores ([09-trials.md](performance/09-trials.md)). Design choices: [ADR-0022](adr/0022-validation-tail-and-training-pipeline.md).
`EPOCHS` and `PATIENCE` override the `epochs=` and `patience=` of the configuration (only those keys of
the generated copy; `PATIENCE=0` turns the early stopping off, and then the best weights are **not**
restored: `network.nn` is the last epoch and the snapshots are the safety net), and
`BATCH` overrides the `batch=` of the generated copy (a larger batch gives the threads more rows per
step but fewer updates per epoch, so the result is not comparable with a run at the configuration's own
batch size; the train line is recorded in `run.txt`; `aws_train.sh` does not pass it on), and
`PROGRESS_EVERY` (default 5) sets the `--progress-every` of the training: the progress goes to
`progress.log`, the checkpoint to `checkpoint.ckpt`, and running the script again with the same run
directory and settings resumes the training (`sh scripts/resume_training.sh [CONFIG [RUN]]` does it in
the background, with batch 384 by default, and refuses to start a run that is already going). Meanwhile `scripts/watch_epochs.sh` (started by
`train_final.sh`) keeps a copy of the checkpoint of every epoch in `RUN/epochs/` as
`checkpoint.<date>-<time>.epochN.ckpt`, with the epoch's measurements in `RUN/epochs.log`: a backup of
the weights and of the training state of every epoch, from which a training can resume. To back up a
run that is already going, run `sh scripts/watch_epochs.sh RUN_DIRECTORY` yourself. To train unattended on a temporary EC2 instance, see
[ADR-0038](adr/0038-unattended-aws-training.md) and the header of `scripts/aws_train.sh`
(`launch`, `status`, `fetch`, `cleanup`), for instance `PROGRESS_EVERY=50 EPOCHS=200 PATIENCE=0 sh scripts/aws_train.sh launch` to reach a snapshot at epoch 50 with `PROGRESS_EVERY=50` (the default period is 5). `MAX_MINUTES` (default: epochs x 40 + 60) makes the instance
shut down, and so terminate, after that time even if the training hangs; it can cut a very long run,
in which case only the uploads made every 10 minutes (progress, snapshots, checkpoint) survive.
