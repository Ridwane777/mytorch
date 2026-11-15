# Architecture

How the project is organized and how data flows through it, without reading the code. The reasons
behind each choice are in the [ADRs](adr/README.md); this page describes what exists.

## 1. Four parts and one dependency rule

| Part | Where | Role |
| --- | --- | --- |
| **`my_torch` library** | `include/my_torch/`, `src/my_torch/` → `build/libmy_torch.a` | A generic neural-network library: matrices, layers, losses, optimizers, training loop, metrics, network file. Knows nothing about chess |
| **Analyzer** | `src/analyzer/` → `my_torch_analyzer` | Reads FEN positions, encodes them, trains or queries a network (`--train`, `--predict`) |
| **Generator** | `src/generator/` → `my_torch_generator` | Writes untrained networks from configuration files |
| **Bonus** | `bonus/` → `bonus/my_torch_oracle`, `bonus/my_torch_positions` (own Makefile) | A rule-based chess oracle, used to check data and to generate labeled positions, never to answer for the network |

```mermaid
flowchart LR
    bonus["bonus/<br/>oracle"] --> analyzer["src/analyzer/<br/>FEN, encoding, CLI"]
    analyzer --> lib["include/my_torch/<br/>library"]
    generator["src/generator/<br/>generator CLI"] --> lib
```

Arrows point from a part to what it uses. The library depends on nothing but the C++ standard
library; the analyzer and the generator only include the public headers of `include/my_torch/`
(the contract between the two work packages, ADR-0004); `bonus/` reuses the analyzer's FEN parser,
and the main code never includes `bonus/`.

## 2. Module map

### Library (`include/my_torch/`, `src/my_torch/`)

| Module | Content | Decision |
| --- | --- | --- |
| `Matrix` | Dense row-major matrix of `double` (`Scalar`): blocked, threaded product (also `transposeMatmul`, `matmulTranspose`), transpose, element-wise operations, bias broadcast, row sums | ADR-0003, ADR-0024 |
| `Parallel` | Persistent thread pool: `parallelFor`, `threadCount`, `setThreadCount` (default: hardware threads or `MY_TORCH_THREADS`) | ADR-0024 |
| `Error` | `my_torch::Error`, the only exception type thrown by the project | ADR-0004 |
| `Layer`, `Dense`, `Activations`, `Dropout`, `Normalization` | The `Layer` interface (`forward`, `backward`, `parameters`, `penalty`, `save`, `saveState`, `outputWidth`, `type`) and its layers: `Dense` (with L1/L2), `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout`, `BatchNorm`, `LayerNorm` | ADR-0004, ADR-0010, ADR-0033 |
| `Convolution` | `ToPlanes`, `Conv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool`: the spatial layers (channel-major rows), `Layer::outputWidth` tells the width each layer needs and gives | ADR-0037, ADR-0041 |
| `Attention` | `ToTokens`, `TokenDense`, `Positional`, `SelfAttention`, `TokenNorm`: the token layers of a mini transformer (rows `tokens × dim`, token-major), blocks built from `Residual` | ADR-0047 |
| `Loss`, `Losses` | The `Loss` interface (`value`, `gradient`, `fusesSoftmax`) and `MSE`, `MAE`, `Huber`, `BinaryCrossEntropy`, `CrossEntropy` | ADR-0007 |
| `Optimizer`, `Optimizers` | The `Optimizer` interface (`step`, `setLearningRate`) and `SGD` (momentum, Nesterov), `Adam`, `AdamW`, `RMSProp`, `Adagrad` | ADR-0008 |
| `Schedulers` | `ConstantRate`, `StepDecay`, `ExponentialDecay`, `CosineAnnealing`, `Warmup` | ADR-0008 |
| `Network` | A sequence of layers: `add`, `forward`, `predict`, `fit`, `parameters`, `save`, `load`, `fromConfig`; `ModelInfo` (encoding, labels, training settings); `DataSource` and `MatrixSource`, where `fit` takes its batches | ADR-0004, ADR-0009, ADR-0021 |
| `Data` | Stratified split, stratified k-fold, `SubsetSource` (a view on some rows of a source), `targetClasses` | ADR-0027 |
| `Metrics` | Confusion matrix (from probabilities or class indices), merge of classes, accuracy, precision, recall, F1, log-loss, top-k, ROC AUC, calibration, CSV/JSON export; every published score goes through it (`scripts/score.cpp`) | ADR-0011, ADR-0026 |
| `Checkpoint` | `saveCheckpoint`, `loadCheckpoint`, `TrainingState`; atomic text file with a checksum and a fingerprint of the training | ADR-0032 |
| `Search` | `SearchSpace`, `Trial`, `SearchStrategy` with `GridSearch` and `RandomSearch`, `SearchResult`; resumable CSV log | ADR-0028 |
| `Factory` | `makeLoss`, `makeOptimizer`, `makeScheduler`: build the training objects from their description in a file | ADR-0012 |
| `Serialization.cpp`, `BinaryFormat.cpp` | Reader and writer of network files (binary, and text for inspection) and configuration files | ADR-0012, ADR-0013, ADR-0042 |

### Analyzer (`src/analyzer/`)

| Module | Content | Decision |
| --- | --- | --- |
| `Fen` | `parseFen`, `parseLine`, `parseFile`: a FEN line into a `Position` (64 squares, side to move, castling, en passant, counters) and an optional `Label` | ADR-0005 |
| `Encoding` | The `Encoding` interface, the table of encodings looked up by name (`fen-onehot-768`, `fen-onehot-773`, `fen-signed-385`, `fen-attacks-901`), and `PositionSource`, which encodes training positions batch by batch | ADR-0014, ADR-0016, ADR-0021 |
| `Cli` | `analyzer::run`: argument parsing, `--predict`, `--train`, `--save`, `--checkpoint`, `--progress-every` | ADR-0014, ADR-0032, ADR-0038 |
| `main.cpp` | Calls `run`, turns any exception into a message on standard error and exit code 84 | ADR-0004 |

### Generator (`src/generator/`) and bonus (`bonus/`)

| Module | Content | Decision |
| --- | --- | --- |
| `Generator` | `generator::run`: checks every configuration, then writes `<name>_1.nn` to `<name>_N.nn` | ADR-0013 |
| `bonus/oracle/Oracle` | `isAttacked`, `legalMoves`, `play`, `classify` (check, checkmate, stalemate, illegal), `perft` | ADR-0006 |
| `bonus/eda` | `my_torch_eda`: one pass over the dataset with the oracle, statistics for the exploratory analysis | - |
| `bonus/positions/Positions` | `generate`: sparse legal positions labeled by the oracle, a chosen share of them stalemates (`bonus/my_torch_positions`) | ADR-0015 |

## 3. Main classes

```mermaid
classDiagram
    class Layer {
        <<interface>>
        forward(input, training) Matrix
        backward(gradOutput) Matrix
        parameters() Parameter[]
        penalty() Scalar
        save(out)
        type() string
    }
    class Loss {
        <<interface>>
        value(prediction, target) Scalar
        gradient(prediction, target) Matrix
        fusesSoftmax() bool
    }
    class Optimizer {
        <<interface>>
        step(parameters)
        setLearningRate(rate)
    }
    class Scheduler {
        <<interface>>
        rate(epoch) Scalar
    }
    class Network {
        add(layer)
        predict(input) Matrix
        fit(train, validation, loss, optimizer, config, scheduler) History
        save(path)
        load(path)$ Network
        fromConfig(path, index)$ Network
    }
    Layer <|-- Dense
    Layer <|-- ReLU
    Layer <|-- LeakyReLU
    Layer <|-- Sigmoid
    Layer <|-- Tanh
    Layer <|-- Softmax
    Layer <|-- Dropout
    Layer <|-- BatchNorm
    Layer <|-- LayerNorm
    Loss <|-- CrossEntropy
    Loss <|-- MSE
    Optimizer <|-- Adam
    Optimizer <|-- SGD
    Network o-- Layer : layers, in order
    Network ..> Loss : uses in fit
    Network ..> Optimizer : uses in fit
    Network ..> Scheduler : uses in fit
```

Only the main subclasses are drawn. Activations and dropout are layers without parameters
(ADR-0004), so a network is a plain list of layers. A `Parameter` pairs a weight matrix with its
gradient, which is how the optimizer reaches the weights without knowing the layers.

## 4. Flows

### Generate: configuration → untrained network

```mermaid
sequenceDiagram
    participant User
    participant Generator as my_torch_generator
    participant Network
    User->>Generator: basic_network.conf 3
    Generator->>Network: fromConfig(conf, 0) for each configuration (check only)
    loop k = 1..3
        Generator->>Network: fromConfig(conf, k-1)
        Network-->>Generator: layers with seeded random weights, ModelInfo
        Generator->>Network: save("basic_network_k.nn")
    end
```

### Train: network file + labeled positions → trained network file

```mermaid
sequenceDiagram
    participant User
    participant Analyzer as my_torch_analyzer
    participant Fen
    participant Encoding
    participant Network
    User->>Analyzer: --train [--save OUT] net.nn train.txt
    Analyzer->>Network: load(net.nn)
    Analyzer->>Fen: parseFile(train.txt, Required)
    Analyzer->>Analyzer: map expected outputs to labels, hold out the validation fraction
    Analyzer->>Network: fit(training source, validation source, loss, optimizer, config, scheduler)
    loop each epoch, each batch
        Network->>Encoding: source.fill(rows of the batch): encode only these positions
        Network->>Network: zero gradients, forward, loss, backward, optimizer step
    end
    Network-->>Analyzer: History (loss, accuracy per epoch)
    Analyzer->>User: one line per epoch
    Analyzer->>Network: save(OUT or net.nn)
```

`fit` takes its examples from a `DataSource` (ADR-0021): the analyzer's `PositionSource` keeps the
parsed positions and encodes only the rows of each batch, so memory does not grow with the encoded
width of the whole file. Inside `fit`, a batch goes forward through the layers, the loss gives the first gradient, and each
layer's `backward` returns the gradient of the layer before it (the final Softmax is skipped when
the loss fuses it, ADR-0007); the optimizer then updates every `Parameter`.

### Predict: network file + positions → one label per line

```mermaid
sequenceDiagram
    participant User
    participant Analyzer as my_torch_analyzer
    participant Fen
    participant Network
    User->>Analyzer: --predict net.nn boards.txt
    Analyzer->>Network: load(net.nn)
    Analyzer->>Fen: parseFile(boards.txt, Optional) (whole file checked first)
    loop 4,096 positions at a time
        Analyzer->>Network: predict(encoded positions)
        Network-->>Analyzer: probabilities
        Analyzer->>User: label of the most probable output, one per line
    end
```

## 5. The network file

A network lives in one binary file (ADR-0042), written by the generator, read and rewritten by the
analyzer, and pushed as `my_torch_network_*.nn` for the evaluation harness. It starts with a signature and
a format version, holds the input encoding name, the labels in order, the training settings and
every layer with its weights (keywords as one-byte codes, integers and reals in binary), and ends
with a checksum. `Network::saveText` writes the same content as text (the format of ADR-0012, which
`load` still reads), and `bonus/my_torch_nn_dump` prints a network file that way. A configuration
file (ADR-0013) is text, written by hand, in the text syntax without the weights.

## 6. Errors

Every error is a `my_torch::Error` with a message naming the file and, when it applies, the line.
It travels up to `main`, which prints `my_torch_analyzer: <message>` (or `my_torch_generator: …`) on
standard error and returns 84. The analyzer and the generator check every input before writing or
printing anything, so an error never leaves a partial output.

## 7. Tests

| Kind | What it guarantees |
| --- | --- |
| Finite differences (`tests/my_torch/gradient_check.hpp`) | Every backward pass, loss gradient, penalty and full-network gradient matches the numerical slope to 1e-6 |
| Hand-computed cases | Layers, metrics and the FEN parser reproduce values worked out by hand |
| Mutation checks | Some tests were run against code broken on purpose (wrong derivative, missing Softmax skip, 15 digits instead of 17, swapped precision and recall) to prove they catch the error |
| End-to-end | The generator and the analyzer are run like the real programs, output compared line by line, every error case exits 84 |
| perft and dataset check | The oracle's move generator matches published move counts; the oracle agrees with all 723,019 dataset labels |

`make tests_run` runs the main suite and `make -C bonus tests_run` the bonus one; `make debug`
rebuilds with AddressSanitizer and UndefinedBehaviorSanitizer.
