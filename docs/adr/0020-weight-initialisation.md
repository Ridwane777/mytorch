# ADR-0020: Weight initialisation chosen per layer in the configuration, original default kept

- **Status**: Accepted
- **Date**: 2026-10-25
- **Deciders**: Meric-BG (author), Ridwane777 (accepted)
- **Issue**: #6

## Context

`Dense` drew its weights from U(±√(1 / fan_in)) and nothing else: a provisional choice (the `Dense`
header announced Xavier and He for #6), and the only value the `init` key of a configuration could take
(ADR-0013, whose ❌ about "only one initialization" this ADR lifts). The variance of the starting
weights decides whether signals and gradients shrink or grow from layer to layer, so it matters more
as the network gets deeper. The default has a variance of 1 / (3·fan_in), a third of LeCun's
1 / fan_in.

Constraints: no library (ADR-0003: the linear algebra is ours, no LAPACK), a generated network must
stay reproducible from the seeds of the configuration (ADR-0013), and **existing configurations must
keep generating the same networks**. The network file stores the weights themselves (ADR-0012), so
an initialisation only matters when a network is generated.

## Options considered

**Where the initialisation is chosen**

| Option | Pros | Cons |
| --- | --- | --- |
| **`init=<name>` on each `dense` line of a configuration (the key ADR-0013 reserved)** | A network can mix initialisations, e.g. He before a ReLU and LeCun before a tanh; nothing new to parse | One word to repeat on each layer |
| One global `init` section for the whole network | Written once | A network mixing `tanh` and `softmax` has no single right choice; collides with `init seed=` |
| Chosen automatically from the activation that follows | Nothing to write | Hidden coupling between two layers; a user cannot override it |

**Where it is kept**

| Option | Pros | Cons |
| --- | --- | --- |
| **Configuration only: the network file is unchanged** | Old files and configurations load as before; no format version bump | A trained file does not say how it was initialised |
| Store `init=` in the network file (ADR-0012 format version 2) | Complete provenance | A new format version for information that has no effect once the weights are written |

**Normal distribution**

| Option | Pros | Cons |
| --- | --- | --- |
| **Untruncated N(0, variance)** | Exactly the target variance; `std::normal_distribution`, nothing to write | Rare weights beyond 3 standard deviations |
| Truncated normal (as Keras does, at 2 standard deviations) | No extreme weight | A rejection loop and a correcting constant to get the target variance back |

**Orthogonal initialisation**

| Option | Pros | Cons |
| --- | --- | --- |
| **Modified Gram-Schmidt on a Gaussian matrix, in house** | A short file-local function, no dependency; orthogonal to better than 1e-9 on the tested shapes (tall, wide, square) | Less stable than Householder on ill-conditioned input, which a random Gaussian matrix is not |
| QR or SVD from LAPACK / Eigen | Reference quality | Forbidden: no linear algebra library (ADR-0003) |
| Householder QR | Numerically the most robust | More code for no measurable gain on these matrices |

**Default**

| Option | Pros | Cons |
| --- | --- | --- |
| **`uniform` stays the default, with the same draws as before** | Existing configurations generate byte-identical networks (checked on `configs/basic_network.conf`) | Not the best known default (see Consequences) |
| Make He or Xavier the default | Better on paper | Silently changes every generated network, and our measurement does not show a winner (below) |

## Decision

`Init` is an `enum class` in `Dense.hpp`, a defaulted last parameter of the `Dense` constructor,
and each `dense` line of a configuration may carry `init=<name>`. The names are `uniform` (the
default, U(±√(1 / fan_in)) as before), `lecun_uniform`, `lecun_normal`, `xavier_uniform`,
`xavier_normal`, `he_uniform`, `he_normal` and `orthogonal`. With fan_in = inputs and
fan_out = outputs, the target variance *v* is 1 / fan_in (LeCun), 2 / (fan_in + fan_out) (Xavier) or
2 / fan_in (He); the uniform forms draw from U(±√(3·v)), the normal forms from an untruncated
N(0, *v*). `orthogonal` fills the larger dimension × smaller dimension matrix with N(0, 1),
orthonormalises its columns by modified Gram-Schmidt (a column that comes out shorter than 1e-12 is
redrawn) and transposes when needed, so that W has orthonormal columns when inputs ≥ outputs and
orthonormal rows otherwise, with a gain of 1. Biases stay at zero. Everything draws from
`std::mt19937(seed)`; an unknown name is a line-numbered error that lists the eight names.

## Consequences

- ✅ Eight initialisations, chosen layer by layer with one word in the configuration; the
  variances, the bounds of the uniform forms and the orthogonality (tall, wide, square, 1e-9) are
  tested, as are the same seed giving the same weights, the zero biases, the unknown name and the
  refusal of `init=` in a network file.
- ✅ Existing configurations, with or without `init=uniform`, generate byte-identical networks; the
  file format and the analyzer do not change.
- ✅ The comparison is measured ([06-initialisation.md](../performance/06-initialisation.md)), on a
  synthetic two-spirals problem with a 2-64-64-64-1 network, 5 seeds, Adam at 0.001: with `tanh`
  the default is the slowest (final loss 0.218 against 0.0632 for `lecun_normal`, after 400 epochs).
- ❌ The default `uniform` keeps its original variance, 1 / (3·fan_in): the measurement above shows
  it is clearly slower than LeCun's on `tanh`. We keep it for compatibility, so a user who wants the
  faster start has to ask for it with `init=`.
- ❌ The same measurement shows **no gain from He on ReLU**: `he_uniform` (0.0374) and `he_normal`
  (0.0642) end above the default (0.00985), and the seeds do not overlap. We have no explanation
  and no chess measurement, so we cannot recommend an initialisation for the chess network yet, and
  `configs/basic_network.conf` is left as it was.
- ❌ `Dense.hpp` is the public contract between the two work packages (AGENTS.md): the new `Init`
  type and constructor parameter need both partners' review. Existing calls compile unchanged.
- ❌ `std::normal_distribution` and `std::uniform_real_distribution` are implementation-defined: the
  normal forms and `orthogonal`, and in principle `uniform` too, can give other weights with
  another standard library. The byte-identity of the default was checked with libstdc++ only (g++
  13.3.0, and g++ 15.2.0 on a second machine, where the benchmark files came out byte-identical). A
  network file is not affected, since it stores its weights. ADR-0009 had passed this fix to #6; it
  is deliberately left out: portable draws change every initial weight, so every committed
  measurement (optimizers, overfitting, stalemate, encodings, class imbalance, initialisation) would
  have to be run again. It is deferred to a follow-up issue, to do before the final network (#17).
- ❌ The untruncated normal forms give a few weights beyond 3 standard deviations; and modified
  Gram-Schmidt would lose orthogonality on ill-conditioned input, which a random matrix is not.
- ❌ A network file does not say which initialisation produced it.
- Follow-up: measure on the real dataset (#17, #18), then choose the `init` of the chess
  configurations from that measurement and, if one is clearly better than `uniform`, write a new
  ADR to change the default.
