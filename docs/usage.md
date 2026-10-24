# Usage: from a configuration to predictions

```bash
make                                              # builds my_torch_analyzer and my_torch_generator

# 1. Write blank networks from a configuration (docs/configuration.md)
./my_torch_generator configs/basic_network.conf 1          # writes basic_network_1.nn

# 2. Train: each line of the file is a FEN followed by its expected output
./my_torch_analyzer --train --save trained.nn basic_network_1.nn train.txt
./my_torch_analyzer --train basic_network_1.nn train.txt   # without --save: overwrites the file

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
- Training settings come from the network file: loss, optimizer, scheduler, epochs, batch size,
  seed, early stopping and the `validation` fraction held out from the training file.
- Every input is checked before anything is printed or written. On an error the message goes to
  standard error and the exit code is 84.

Design choices: [ADR-0014](adr/0014-analyzer-command-line.md).
