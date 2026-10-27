# ADR-0012: Network file format, versioned self-describing text with a checksum

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #8

## Context

The network file is the only link between the generator (`my_torch_generator` writes untrained
networks), the analyzer (`--train` loads, trains and saves; `--predict` loads and predicts) and the
autograder, which loads our pushed `my_torch_network_*.nn`. The subject leaves the format free but
asks to think about what information is actually necessary. Three facts drive the choice:

- the analyzer must answer exactly like the original network with only this file: it needs the
  architecture, the weights, how to encode an input and which output is which label;
- `--train` takes no hyperparameters on its command line, so the training settings must travel with
  the network;
- a file that loads without error but with wrong values (truncated, a digit changed, outputs in
  another order) gives wrong answers silently, which is worse than an error.

## Options considered

**Encoding of the file**

| Option | Pros | Cons |
| --- | --- | --- |
| **Text, numbers with 17 significant digits** | Readable and editable by hand, diffable in Git, shown as is in a defense; 17 digits read back every double exactly | 2.7 times larger than raw binary, slower to parse |
| Text, numbers in hexadecimal floating point | Exact and fast to parse | Numbers unreadable for a human |
| Binary | Smallest and fastest | Unreadable; byte order and layout to specify; harder to debug |

**What goes in the file**

| Option | Pros | Cons |
| --- | --- | --- |
| **Layers and weights, plus input encoding, label names in order and training settings** | Self-sufficient: `--predict` and `--train` need nothing else | The settings are descriptions (type and key=value) that the factories must turn into objects |
| Layers and weights only | Smallest | Label order and encoding live in the code, so changing the code changes the answers silently; `--train` needs its settings elsewhere |

**How layers are written and read**

| Option | Pros | Cons |
| --- | --- | --- |
| **Each layer writes itself (`Layer::save`); a table maps a type to its reader** | A new layer (convolution, normalization) adds its `save` and one table entry | One more virtual method on `Layer` |
| One central function that knows every layer | Everything in one place | Grows with every layer and reaches into their internals |

**Protection against corruption**

| Option | Pros | Cons |
| --- | --- | --- |
| **Strict parsing plus an FNV-1a 64 checksum on the `end` line** | Catches truncation, bad numbers, inconsistent sizes, and a changed digit that still parses | Accidental corruption only, not tampering; a hand-edited file must drop the checksum |
| Strict parsing only | Simpler | A changed digit loads without error |

## Decision

A network file is text, version 1. In order: the line `my_torch 1`; the input encoding name and
width; the label names, one per line; the loss, optimizer and scheduler as `type key=value` lines;
the training configuration; the layers, each written by its own `save` (a Dense layer writes one
line of weights per input and one line of biases, numbers with 17 significant digits); and
`end checksum=<FNV-1a 64>`, computed over every byte before that line. A file ending with a plain
`end` is accepted without the check, for files written by hand. `load` checks the signature, the
version, every keyword, every number (finite), every count, the layer settings (through the layer
constructors), the consistency of the widths from the input to the number of labels, the checksum
and that nothing follows `end`; every error names the file and the line. The training settings are
stored as `ComponentSpec` descriptions and turned into objects by `Factory.hpp`, which rejects
unknown types and keys.

## Consequences

- ✅ Save then load gives bit-identical predictions for a network with every layer type, and saving
  a loaded network gives the same file byte for byte (tested); with 15 digits instead of 17, three
  tests fail.
- ✅ Training 10 epochs, saving, loading and training 10 more gives the same weights as 20 epochs
  in one go, with plain SGD (tested).
- ✅ Each kind of corrupted file is rejected with its line number, including a single changed digit
  in a weight, caught by the checksum (tested); the analyzer turns these errors into exit code 84 (#15).
- ✅ For the planned 768-128-5 network: 2.15 MB, saved in 35 ms and loaded in 16 ms
  (`docs/performance/13-serialization.md`).
- ❌ 2.7 times larger than a raw binary format; at about 5 million parameters a file would pass
  GitHub's 100 MB limit, and a binary version 2 would be needed.
- ❌ The optimizer state (Adam's moments) is not saved: resuming with Adam is not identical to an
  uninterrupted run. Checkpoints (#35) will add it in a version 2.
- ❌ A reloaded dropout layer restarts its mask sequence from its seed.
- ❌ `Layer.hpp`, `Dense.hpp`, `Dropout.hpp`, `Activations.hpp` and `Network.hpp` change; version 1
  is now a compatibility commitment, and any change of format needs a new version number.
- Follow-up: the generator (#9) builds `ModelInfo` from its configuration file; the analyzer (#15)
  builds the loss, optimizer and scheduler with `Factory.hpp` for `--train`, and checks that it
  knows the encoding named in the file.
