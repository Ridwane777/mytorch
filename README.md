# my_torch — hand-written neural network and chessboard analyzer

Project in C++20 ([ADR-0002](docs/adr/0002-cpp-language.md)). Two deliverables:

- **`my_torch`** (`include/my_torch/`, `src/my_torch/`): a generic neural-network library, written
  without any deep learning library (PyTorch, TensorFlow… are forbidden). It can create, train,
  save and reload a network.
- **`my_torch_analyzer`**: an executable that reads chess positions in FEN notation and predicts
  the state of the game (`Checkmate`, `Check`, `Nothing`, with the color), by supervised learning.

> Status (2026-10-24): the library is complete enough to train and save a network (dense layers,
> activations, losses, optimizers and schedulers, mini-batch training, regularization, dropout,
> early stopping, metrics, network file). The generator, the analyzer (`--train`, `--predict`)
> with a baseline input encoding, the FEN parser and a rule-based oracle are done. Still to do:
> the encoding variants (#14), training on the full dataset and the final network (#17). The
> technical choices are in [docs/adr/](docs/adr/README.md).

## Build and usage

Requirements: g++ (C++20) and make; Criterion for the tests.

```bash
make                    # my_torch_analyzer and my_torch_generator
./my_torch_analyzer --predict my_torch_network_final.nn chessboards.txt
./my_torch_analyzer --train [--save NEW.nn] NETWORK.nn chessboards.txt
./my_torch_generator configs/basic_network.conf 3   # writes basic_network_1.nn to _3.nn
```

On error, the message goes to standard error and the program exits with code 84. Step by step:
[docs/usage.md](docs/usage.md).

| Rule | Effect |
| --- | --- |
| `make` | Builds with `-O2` |
| `make debug` | Rebuilds with AddressSanitizer and UBSan |
| `make tests_run` | Builds and runs the Criterion tests (with coverage) |
| `make clean` / `fclean` / `re` | Cleaning and rebuilding |

## Repository layout

| Path | Content |
| --- | --- |
| `include/my_torch/` | Public headers of the library: the only contract between the module and the analyzer |
| `src/my_torch/` | Library implementation, compiled into `build/libmy_torch.a` |
| `src/analyzer/` | Chess part: FEN reading, encoding, command-line interface |
| `src/generator/` | `my_torch_generator`: creates blank networks from a configuration |
| `tests/` | Criterion tests, same subfolders as `src/` |
| `my_torch_network_final.nn` | Pre-trained network evaluated by the autograder: 96.2 % accuracy on 71,555 unseen positions ([measure](docs/performance/10-final-network.md)) |
| `configs/` | Generator configurations ([format](docs/configuration.md)) |
| `scripts/` | Training (a backup of every epoch in `runs/<name>/epochs/`), unattended training on AWS, benchmarks and graphs, to reproduce our results |
| `bonus/` | Bonuses, with their own Makefile (required by the subject): the rule-based chess oracle, the stalemate position generator and a chess GUI that analyses each move with the network ([guide](docs/gui.md)) |
| `docs/` | Project file: decisions, performance, legal, business, retrospective. Map: [docs/README.md](docs/README.md) |

The datasets (`dataset/`) are not pushed: they are too heavy. They are shown at the defense and
described in `docs/performance/`.

Before the first commit: [CONTRIBUTING.md](CONTRIBUTING.md).
