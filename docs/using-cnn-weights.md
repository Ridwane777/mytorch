# Using the CNN weights (epoch 10 snapshot)

`bonus/networks/my_torch_network_cnn_epoch10.nn` holds the weights of `configs/cnn_network.conf` after **10 of at most
30 epochs** (early stopping, patience 3) of the full-dataset training (`scripts/train_final.sh`, batch 384, seed 42). The
training was still running when it was saved: this is an **intermediate snapshot**, not the final
network. It is the file `runs/cnn_full/network.nn.20261102-171758.epoch10.nn` written by
`--progress-every 5` (ADR-0038), copied here because `/runs/` is not pushed.

## Predict with it

```sh
make                                              # builds ./my_torch_analyzer
cut -d' ' -f1-6 positions.txt > boards.txt        # a FEN per line (the first six fields)
./my_torch_analyzer --predict bonus/networks/my_torch_network_cnn_epoch10.nn boards.txt
```

One line per position, in file order: `Nothing`, `Check White`, `Check Black`, `Checkmate White` or
`Checkmate Black`. Errors go to stderr with exit code 84. The prediction takes about 50 s for 71,555
positions with `MY_TORCH_THREADS=2` (one run, measured while another training used the other cores).

## Score it

```sh
g++ -std=c++20 -O2 -pthread -Iinclude scripts/score.cpp build/libmy_torch.a -o score
./my_torch_analyzer --predict bonus/networks/my_torch_network_cnn_epoch10.nn boards.txt > predictions.txt
./score bonus/networks/my_torch_network_cnn_epoch10.nn positions_with_labels.txt predictions.txt OUT_DIRECTORY
cat OUT_DIRECTORY/scores.txt
```

## Continue training from it

Fine-tune with `./my_torch_analyzer --train --save NEW.nn bonus/networks/my_torch_network_cnn_epoch10.nn TRAINING_FILE`.
The training settings come from the `train` line of the `.nn` file: 30 epochs at most, batch 384, early
stopping with patience 3, and the **last 71,558 lines** of TRAINING_FILE held out as validation. So
TRAINING_FILE needs more than 71,558 lines (otherwise the command fails), or edit that line first.
This restarts the optimizer (Adam moments are not in a `.nn` file) and the epoch counter. To resume
exactly, use the `checkpoint.<date>-<time>.epoch10.ckpt` backup of `RUN/epochs/` instead (see
`scripts/resume_training.sh`).

## Measured results

On the 71,555 positions of the test split (`runs/cnn_full/test.txt`, never used for training or for
choosing anything), measured once with `scripts/score.cpp`:

| Metric | Value |
|---|---|
| Accuracy | 98.07 % |
| Macro F1 | 0.9729 |
| Nothing | recall 1.0000, precision 0.9999 |
| Check White / Black | recall 0.9564 / 0.9624 |
| Checkmate White / Black | recall 0.9776 / 0.9794 |

During training, epoch 10 had validation accuracy 98.06 % (loss 0.0512), consistent with the test
score.

## Limits

- Intermediate snapshot: the final network will replace it, and its numbers will differ.
- The errors are almost all a Check read as a Checkmate of the same colour (587 White, 424 Black) or
  the reverse (207 White, 160 Black); `Nothing` is almost never confused: 1 of the 29,756 `Nothing` positions was misread, and 2 Check White
  positions were predicted `Nothing`.
- Only the test split above was measured; no other dataset was tried.
