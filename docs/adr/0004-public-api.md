# ADR-0004: Public API of the my_torch library

- **Status**: Accepted
- **Date**: 2026-10-21
- **Deciders**: Ridwane777, Meric-BG
- **Issue**: #1

## Context

The pair codes in parallel: layers and backpropagation (#2, #4) on one side, losses, optimizers and
saving (#3, #5, #8) on the other. The shared signatures, written in `include/my_torch/`, must be fixed
early so that the pieces fit together. The API must also accept the planned extensions without being
rewritten: convolution (#29), normalization (#30), dropout (#7), new optimizers.

## Options considered

**Where to put the activation functions (ReLU, Softmax…)?**

| Option | Pros | Cons |
| --- | --- | --- |
| **A layer in its own right, with no weights** | A network is a simple list of layers; each activation has its own testable forward/backward; same model as PyTorch's `nn.ReLU` | More layers in the list; one more intermediate result in memory |
| A parameter of the dense layer | Fewer layers; slightly less memory | More complex dense layer; every new layer (convolution…) must reimplement the activations |

**Shape of the network**

| Option | Pros | Cons |
| --- | --- | --- |
| **Sequential** (a queue of layers) | Simple to write, train and save; enough for the whole mandatory part | Does not allow residual blocks |
| Graph of layers | Allows modern architectures (residuals, attention) | Much more complex (computation order, saving) for an optional need |

## Decision

- **`Layer`** (abstract class): `forward(input, training)` computes the output; `backward(gradOutput)`
  receives dL/doutput, accumulates the gradients of its weights and returns dL/dinput (chain
  rule); `parameters()` exposes its (weight, gradient) pairs; `type()` gives a stable name for
  saving. `training` distinguishes training from prediction (dropout, normalization).
- **Activations are layers** with no parameter.
- **`Loss`**: `value` (mean loss of the batch) and `gradient` (first gradient of backpropagation).
  Softmax and cross-entropy are combined to get the simplified gradient `prediction − target`,
  simpler and more numerically stable.
- **`Optimizer`**: `step(parameters)` updates the weights without knowing the layers;
  `setLearningRate` lets schedulers vary the rate.
- Sequential **`Network`**: owns its layers (`std::unique_ptr`), `predict`, `fit` (epochs,
  mini-batches, shuffling, validation, resetting gradients to zero after each step), `save` / `load`.
  `fit` returns a **`History`** (one `EpochStats` per epoch) to plot the learning curves.
- **Fixed seed** (`TrainConfig::seed`) for all randomness: the same training gives back the same
  network.
- **`my_torch::Error`** for every error; the executables catch it and exit with 84.

Target usage example:

```cpp
Network net;
net.add(std::make_unique<Dense>(768, 128));
net.add(std::make_unique<ReLU>());
net.add(std::make_unique<Dense>(128, 5));
net.add(std::make_unique<Softmax>());

CrossEntropy loss;
Adam optimizer(0.001);
History history = net.fit(train, &validation, loss, optimizer, {.epochs = 20, .batchSize = 32});
net.save("my_torch_network_full.nn");
```

## Consequences

- ✅ Both work packages code in parallel against the same headers.
- ✅ A new layer or a new optimizer is added without changing the rest (polymorphism).
- ✅ Reproducible trainings, a project requirement and needed for comparisons.
- ❌ No residual blocks or attention with a sequential network: #31 and #32 will require a new
  ADR to move to a graph.
- ❌ The Softmax + cross-entropy coupling forces this case to be handled separately in
  backpropagation.
- ❌ One activation per layer costs one more intermediate result in memory.
- Follow-up: any change to the headers in `include/my_torch/` goes through a PR reviewed by both.
