# ADR-0013: Generator configuration in the network file syntax, seeded and checked before writing

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #9

## Context

The subject requires `./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]` to write
`nb` blank networks per configuration (`basic_network.conf 3` gives `basic_network_1.nn` to
`basic_network_3.nn`), and leaves the configuration format free. The network file (ADR-0012)
already describes everything a network needs except its weights. Three more points must be
settled: how the networks of one command differ, where the files go, and what happens when one
argument out of several is wrong.

## Options considered

**Configuration format**

| Option | Pros | Cons |
| --- | --- | --- |
| **The network file syntax without weights, plus comments, optional sections and `init seed=…`** | One syntax to learn, document and defend; the reader and every check of ADR-0012 are reused, with the same line-numbered errors | The reader gains a configuration mode |
| JSON | Widely known | A JSON parser to write or a dependency to add; a second syntax |
| Flat `key=value` (`nb_layers=3`, `layer_1=128`…) | Very simple | Poor fit for a list of layers with their own settings; a third syntax |

**Initial weights**

| Option | Pros | Cons |
| --- | --- | --- |
| **Seeds derived from `init seed` + network index, mixed with SplitMix64 per layer** | Networks of one command differ; the same command writes the same files byte for byte | Changing the seed of one network means changing the configuration |
| Seed from the clock | Different every run | Not reproducible, against the subject |

**Writing**

| Option | Pros | Cons |
| --- | --- | --- |
| **Check every configuration first, then write** | A wrong argument writes nothing, so no half-finished set of files | Each configuration is read twice |
| Write as soon as each configuration is read | Read once | `a.conf 2 bad.conf 1` leaves `a_1.nn` and `a_2.nn` behind with exit code 84 |

Files are written in the current directory, named after the configuration file without its
directory and extension, as in the subject's example.

## Decision

A configuration is the network file syntax without the header, the weights and the checksum; it
accepts `#` comments and blank lines, and its `loss`, `optimizer`, `scheduler`, `train` and `init`
sections may come in any order or be omitted (defaults apply). `Network::fromConfig(path, index)`
builds the network; weights and dropout seeds come from SplitMix64 mixes of `init seed + index` and
the layer index. Reading a configuration or a network file also builds the loss, optimizer and
scheduler once, so that a typo is reported at generation time. The generator checks every
configuration before writing any file.

## Consequences

- ✅ `basic_network.conf 3` writes exactly `basic_network_1.nn` to `_3.nn`; the networks differ and
  a second run writes identical files (tested).
- ✅ Every written network loads back with its configuration (tested); a wrong count, a missing or
  invalid configuration exits 84 and writes nothing, even when only the second configuration is
  wrong (tested).
- ✅ Configuration errors name the line: unknown keyword, duplicated section, unknown layer, weights
  given by mistake, unknown optimizer, inconsistent sizes (tested).
- ❌ Only one initialization exists (`init=uniform`) until #6 adds Xavier and He; the key is
  reserved.
- ❌ Settings written as decimals are stored with 17 significant digits, so `l2=0.0003` reappears
  as `l2=0.00029999999999999997` in the network file: exact, but less readable.
- ❌ Output files are always written in the current directory, with no option to choose another.
- Follow-up: the analyzer (#15) uses the training settings stored by the generator.
