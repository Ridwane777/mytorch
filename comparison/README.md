# comparison/: reference implementations (ADR-0036, ADR-0040)

Everything under this directory may use PyTorch, which the subject forbids in the project. It is
never built, called or tested by the Makefile, and `scripts/check_comparison_isolation.sh` fails if
a machine-learning library is imported anywhere else. Removing `comparison/` changes nothing in the
build, the tests or the pushed network.

| File | Role |
| --- | --- |
| `requirements.txt` | the two packages (torch CPU wheel, numpy) and how to install them |
| `mytorch_net.py` | reads a my_torch configuration (`configs/*.conf`) and builds the same architecture in PyTorch; reads the binary files written by `scripts/pytorch_reference.cpp` |
| `gradient_check.py` | our gradients against autograd on one batch with identical weights (criterion 1e-5) |
| `train_reference.py` | trains the PyTorch twin on the same data, same optimizer, batch size and epochs |

The my_torch side, plot and orchestration are in `scripts/` (`pytorch_reference.cpp`,
`bench_pytorch_reference.sh`, `plot_pytorch_reference.py`): they include no PyTorch header and
import no PyTorch module. Results: [`docs/performance/22-pytorch-reference.md`](../docs/performance/22-pytorch-reference.md).

## Setup

```bash
python3 -m venv comparison/.venv        # git-ignored
comparison/.venv/bin/pip install -r comparison/requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## Run

```bash
make mytorch                              # the library the helper links against
sh scripts/bench_pytorch_reference.sh     # gradient check, thread sweep, training runs, tables and graphs
STEPS=gradient sh scripts/bench_pytorch_reference.sh   # only the gradient check (seconds)
```

The script documents its environment variables (budget, seeds, threads, directories). Large
intermediate files (encoded data, weights, row orders) go to `runs/pytorch_reference/`
(git-ignored); the CSV files and graphs go to `docs/performance/data/pytorch_reference/`.
