# ADR-0042: Binary network file format

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-11-04
- **Deciders**: Ridwane777, after the first defense; accepted by the pair
- **Supersedes**: ADR-0012 (network file format, versioned self-describing text with a checksum)

## Context
ADR-0012 chose a text file: readable, diffable, shown as is in a defense, at 2.7 times the size of
raw binary (docs/performance/13-serialization.md). The first defense judged otherwise: "the whole
file is plain text, without any encoding" cost a point. A network file is data written and read by
programs; the readability argument belongs to a tool, not to the file itself.

What must not change: the analyzer's command line and answers, the content of the file (encoding
name, labels, training settings, layers and their exact weights), the checksum, and the ability to
read the files already written (the pushed network, the CNN snapshot, the test fixtures).

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Raw binary: header, then each layer as fixed fields and arrays | Smallest; fastest to read | One binary reader and writer per layer type (twelve), a second serialization next to the text one, every future layer twice |
| **Typed binary tokens over the same records (chosen)** | Every value binary (keywords as one-byte codes, integers in LEB128, reals as binary64, lengths in front of strings); one encoder and one decoder for every layer, present and future; the text reader and all its checks are reused, so binary and text cannot disagree | One byte of type per token and a byte per record over raw binary; reading decodes to records then parses them, slower than a dedicated reader |
| Text compressed with a general compressor (zlib) | Small; little code | A compression library is a new dependency, or a compressor to write; the content is still text inside |
| Keep text, document it better | No work | Does not answer the defense |

## Decision
`Network::save` writes version 2: the signature `MYTORCH\0`, the version (uint32), the payload
length (uint64), the payload, and an FNV-1a 64 checksum of the payload (uint64), all little-endian.
The payload holds the records of ADR-0012 (input, labels, loss, optimizer, scheduler, train,
layers), each a sequence of typed tokens: word (a code in a fixed, append-only table of the
library's keywords), string (length and UTF-8 bytes, for the names the library does not know:
encodings and labels), integer (LEB128), reals (count and binary64 values: a whole row of weights is
one token), setting (`key=value`), line (a record kept byte for byte, for a label with odd
spacing). Every double is written as its 8 bytes: a file read back gives the same values bit for
bit. Details in `src/my_torch/BinaryFormat.hpp`.

`Network::load` tells the formats apart by the first bytes and reads both. `Network::saveText` writes
the text format, and `bonus/my_torch_nn_dump` prints any network file as text. Configuration files
stay text: they are written by hand. Checkpoints (ADR-0032) keep their text format.

## Consequences
- ✅ No record of a network file is text any more; numbers are binary and exact.
- ✅ 2.55 times smaller for the pushed network (5.4 MB to 2.1 MB, 0.4 % above raw binary64),
  measured in docs/performance/24-binary-format.md.
- ✅ The analyzer, the generator and every program using the library change nothing: same calls,
  same answers (the predictions of the text and binary versions of the pushed networks are compared
  on the whole test split).
- ✅ Old text files still load; a truncated, corrupted or newer file is refused with a message.
- ❌ A network file can no longer be read or diffed in Git directly: a tool is needed
  (`my_torch_nn_dump`).
- ❌ Saving and loading are about 2 times slower than with text (137 ms against 63 ms to load the
  pushed network): the decoder goes back through text records, so every number is formatted and
  parsed again. Still about 6 % of the 2.3 s of a prediction of the test split.
- ❌ Two formats to maintain in the reader, and a keyword table whose order must never change.
- Follow-up: the pushed networks are converted to binary; the checkpoint format could follow the
  same way if its size becomes a problem.
