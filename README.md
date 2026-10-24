# my_torch — hand-written neural network and chessboard analyzer

Project in C++20 ([ADR-0002](docs/adr/0002-cpp-language.md)). Two deliverables:

- **`my_torch`** (`include/my_torch/`, `src/my_torch/`): a generic neural-network library, written
  without any deep learning library (PyTorch, TensorFlow… are forbidden). It can create, train,
  save and reload a network.
- **`my_torch_analyzer`**: an executable that reads chess positions in FEN notation and predicts
  the state of the game (`Checkmate`, `Check`, `Nothing`, with the color), by supervised learning.

> Status (2026-10-24): the library is complete enough to train and save a network (dense layers,
> activations, losses, optimizers and schedulers, mini-batch training, regularization, dropout,
> early stopping, metrics, network file). The generator, the FEN parser and a rule-based oracle
> are done. Still to do: input encoding, the analyzer command line, and the trained network. The
> technical choices are in [docs/adr/](docs/adr/README.md).

## Build and usage

Requirements: g++ (C++20) and make; Criterion for the tests.

```bash
make                    # my_torch_analyzer and my_torch_generator
./my_torch_analyzer --predict my_torch_network_<name>.nn chessboards.txt
./my_torch_analyzer --train [--save NEW.nn] NETWORK.nn chessboards.txt
./my_torch_generator configs/basic_network.conf 3   # writes basic_network_1.nn to _3.nn
```

On error, the message goes to standard error and the program exits with code 84.

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
| `my_torch_network_*.nn` | Pre-trained network evaluated by the autograder |
| `configs/` | Generator configurations ([format](docs/configuration.md)) |
| `scripts/` | Training, benchmarks and graphs, to reproduce our results |
| `bonus/` | Bonuses, with their own Makefile (required by the subject): the rule-based chess oracle |
| `docs/` | Project file: decisions, performance, legal, business, retrospective. Map: [docs/README.md](docs/README.md) |

The datasets (`dataset/`) are not pushed: they are too heavy. They are shown at the defense and
described in `docs/performance/`.

Before the first commit: [CONTRIBUTING.md](CONTRIBUTING.md).
