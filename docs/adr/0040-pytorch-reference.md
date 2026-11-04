# ADR-0040: PyTorch reference: same weights, same batches, same gradients

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-11-04
- **Deciders**: Meric-Gbemetonou (issue #42), approved by Ridwane777; use of the libraries in `bonus/comparaison/` approved by the teaching staff as a bonus beyond the subject

## Context
ADR-0036 allows PyTorch under `bonus/comparaison/` only. Issue #42 asks what to build with it: the same
architectures (the dense network and the CNN), the same hyperparameters, and three checks: our
gradients against autograd on the same initial weights (identical within 1e-5 on one batch), time
per epoch and memory, and an explanation of any accuracy gap.

"Same network, same hyperparameters" does not make two trainings comparable by itself. My_torch
draws its weights with a C++ `std::mt19937`, shuffles with its own Fisher-Yates over a
`std::mt19937_64` (ADR-0009) and encodes the positions in C++; PyTorch has other generators, another
default initialisation and no chess encoding. If the two sides differ in everything, a gap in
accuracy cannot be attributed to anything. The decision is how to make them differ in as few things
as possible, and to measure what remains.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Re-implement our generators, our initialisation and our encoding in Python | Nothing exchanged between the two programs | Three reimplementations to keep identical to C++ (mt19937 distributions are implementation-defined for `normal_distribution`); a mistake would be invisible: the two sides would differ and nobody would know why |
| Compare the two libraries with their own defaults only | Simple; shows what a user of each gets | A gap cannot be explained: initialisation, shuffling, rounding and the algorithm are mixed |
| **Replay: our program writes what it used, PyTorch reads it (chosen)** | PyTorch starts from our exact weights and sees our exact batches, so what is left is the arithmetic; the same C++ encoder feeds both | A small C++ helper and binary files; the replay is only possible for the first seed |
| Check gradients with finite differences only (already done by the unit tests) | No second library | Not independent of our own formulas: autograd is computed by other code from the forward pass alone |

## Decision
1. **Architecture from the configuration.** `bonus/comparaison/mytorch_net.py` reads `configs/final_network.conf`
   and `configs/cnn_network.conf` and builds the PyTorch network from their `layers` lines
   (`dense`, `relu`, `board_planes`, `conv2d`, `max_pool2d`, final `softmax` left to the loss). Nothing
   is typed twice.
2. **A helper on our side, library headers only.** `scripts/pytorch_reference.cpp` (same status as
   the other programs of `scripts/`: it includes `include/my_torch/` and the analyzer's position
   reader, never a PyTorch header) has three commands: `export` (the encoded positions as bytes,
   0 or 1, checked), `gradient` (the weights and the gradients of one batch, computed as
   `Network::fit` does), `train` (the initial weights, the order of the rows of every epoch, the
   final weights, per-epoch measurements). Files are raw little-endian binary read by numpy.
3. **Gradient check.** One batch of 64 positions, the weights loaded in PyTorch, autograd against
   our `backward`. Criterion: for every parameter tensor, the maximum absolute difference and that
   difference divided by the largest absolute gradient of the tensor are both at most 1e-5; the
   loss and the output probabilities are held to the same bound. Done in float64 (our `Scalar`,
   gating) at the initial weights and after 100 Adam steps (non-zero biases, ReLUs in both states),
   and in float32 for information.
4. **Four PyTorch variants against ours**, same data, same optimizer (Adam, same learning rate,
   betas and epsilon), same batch size and number of epochs, scored on the whole validation split
   (the test split is not read by this work):
   - *shared*, float64: our initial weights and our batch order replayed. The difference to our
     result is arithmetic only;
   - *native*, float64: PyTorch's initialisation and shuffling. The difference to our result is the
     starting point and the order, i.e. the seed-to-seed spread plus the initialisation scheme
     (listed in the document);
   - *native*, float32: PyTorch's default precision, the realistic cost.
   - *scheme*, float64: our initialisation distributions (dense U(±sqrt(1 / fan_in)), convolution
     He normal, zero biases) drawn by PyTorch's generator, PyTorch's shuffling. It separates the
     initialisation scheme from the seed: the difference between *native* and *scheme* is the scheme
     (two things at once: convolution weight variance and biases), the one between *scheme* and our
     result is the draws;
   Several seeds for our side and for the native and scheme float64 variants, to say whether a gap is larger
   than the spread between seeds.
5. **Benchmarks on one machine.** Time per epoch (training pass only, scoring excluded, on both
   sides) and the peak memory of the whole process (GNU time), each implementation at its best
   thread count from a short sweep, the machine's load recorded with every run.

## Consequences
- ✅ If the shared variant ends where we end (weights, losses, predictions), our training loop,
  Adam, cross-entropy and every layer including the convolution are right, to rounding, which
  neither a unit test nor a gradient check alone says.
- ✅ The accuracy gap is split into parts that are measured (arithmetic, initialisation and order,
  precision) instead of argued.
- ✅ The gradient check is independent of our own formulas and runs in seconds
  (`STEPS=gradient sh scripts/bench_pytorch_reference.sh`).
- ❌ The replay shares our encoder: an error in the encoding (ADR-0016) would be present on both
  sides and is not detected here.
- ❌ PyTorch's float64 convolution is not its fast path (float32 is): float64 is used for the
  equivalence, and the float32 run is reported next to it. The timings of the float64 variants
  are those of a configuration nobody trains with.
- ❌ Memory is not strictly comparable: PyTorch holds the encoded data as one byte array (rows × 901
  bytes) and the Python runtime, we encode batch by batch (ADR-0021). The document reports the
  baseline of the Python process and the size of the data.
- ❌ The measurements are those of a laptop shared with a desktop session: timings carry the load
  at the time (recorded, not removed).
- ❌ A second check of our code that only covers the two architectures of the configurations
  (dense, ReLU, softmax; Conv2D, max pooling, board_planes); dropout and normalization layers are
  not reproduced.
- Follow-up: moves to Accepted with ADR-0036; re-run (`sh scripts/bench_pytorch_reference.sh`)
  when the PyTorch version, the machine or a layer of the library changes; extend the twin when a
  layer is added to the configurations.
