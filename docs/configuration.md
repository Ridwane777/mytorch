# Configuration files for `my_torch_generator`

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

For each pair, the generator writes `nb` untrained networks named after the configuration file:
`configs/basic_network.conf 3` writes `basic_network_1.nn`, `basic_network_2.nn` and
`basic_network_3.nn` in the current directory. Every configuration is checked before any file is
written: on an error nothing is written, the message goes to standard error and the exit code is
84. `--help` prints the usage. Design choices: [ADR-0013](adr/0013-configuration-format.md).

## Syntax

A configuration is a [network file](adr/0012-network-file-format.md) without weights:

- the same keywords and the same `type key=value` settings;
- no `my_torch 1` header and no `end checksum=…` line;
- lines starting with `#` (after optional spaces) are comments; blank lines are ignored;
- `loss`, `optimizer`, `scheduler`, `train` and `init` may appear in any order, at most once each,
  between `labels` and `layers`; when one is missing its defaults apply.

Example (`configs/basic_network.conf`):

```
# Basic chess network: one hidden layer.
input fen-onehot-768 768
labels 5
Nothing
Check White
Check Black
Checkmate White
Checkmate Black

loss cross_entropy
optimizer adam lr=0.001
scheduler none
train epochs=20 batch=64 seed=42 patience=5
init seed=1

layers 4
dense 768 128 l2=0.0003
relu
dense 128 5
softmax
```

## Reference

### Model

| Line | Meaning |
| --- | --- |
| `input <encoding> <width>` | Name of the input encoding (one word) and number of inputs. Required |
| `labels <count>` then one label per line | Name of each output, in order; labels may contain spaces. Required |
| `layers <count>` then one layer per line | The layers, in order. Required, last section |

### Layers

| Layer | Settings (default) |
| --- | --- |
| `dense <inputs> <outputs>` | `l1` (0), `l2` (0): weight penalty; `init` (`uniform`): only the provisional uniform initialization exists until #6 |
| `relu`, `sigmoid`, `tanh`, `softmax` | none |
| `leaky_relu` | `slope` (0.01) |
| `dropout` | `rate` (required, 0 ≤ rate < 1), `seed` (derived from the network seed when omitted) |

The first `dense` takes `<width>` inputs, each `dense` takes the outputs of the previous one, and
the last one has one output per label; the generator refuses any other combination.

### Training (stored in the network, used by `my_torch_analyzer --train`)

| Line | Types and settings (default) |
| --- | --- |
| `loss <type>` (`cross_entropy`) | `mse`, `mae`, `huber` (`delta` 1), `binary_cross_entropy`, `cross_entropy` (`smoothing` 0, `weights` w1,w2,… one per label) |
| `optimizer <type>` (`adam`) | `sgd` (`lr` 0.01, `momentum` 0, `nesterov` 0/1), `adam` (`lr` 0.001, `beta1` 0.9, `beta2` 0.999, `epsilon` 1e-8, `amsgrad` 0/1), `adamw` (same as adam plus `weight_decay` 0.01), `rmsprop` (`lr` 0.001, `decay` 0.9, `epsilon` 1e-8), `adagrad` (`lr` 0.01, `epsilon` 1e-10) |
| `scheduler <type>` (`none`) | `constant` (`base`), `step` (`base`, `step` 10, `gamma` 0.1), `exponential` (`base`, `gamma` 0.95), `cosine` (`base`, `epochs` 10, `min` 0); each accepts `warmup` (epochs); `base` defaults to 0.001 |
| `train` | `epochs` (10), `batch` (32), `seed` (42), `shuffle` (1), `patience` (0 = no early stopping), `min_delta` (0), `restore_best` (1) |

### Generation

| Line | Meaning |
| --- | --- |
| `init seed=<n>` (1) | Network number *k* (1-based) draws its weights from seeds derived from *n + k − 1*: the networks of one command differ, and the same command always writes the same files |
