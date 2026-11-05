#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Binary network file, format version 2 (ADR-0042). Internal to the library: Network::save writes
// it and Network::load reads it, the public API does not change.
//
//   bytes 0-7    "MYTORCH" and a zero byte
//   bytes 8-11   format version, uint32 little-endian (2)
//   bytes 12-19  payload length in bytes, uint64 little-endian
//   payload      the records of the network, see below
//   last 8       FNV-1a 64 of the payload, uint64 little-endian
//
// The payload holds the same records as the text format of ADR-0012, from "input" to the last
// layer, each one a sequence of typed tokens ended by a zero byte:
//   0x01 word       uint8: index in a fixed table of the library's keywords (layer types, losses,
//                   optimizers, setting names); the table only grows, so old files stay readable
//   0x02 string     uint32 length, then the bytes (UTF-8): names the library does not know, such
//                   as the input encoding and the labels
//   0x03 integer    unsigned LEB128 (7 bits per byte): sizes, counts, seeds, epochs
//   0x04 reals      uint32 count, then count IEEE-754 binary64 little-endian: weights, biases,
//                   learning rates; a run of consecutive values is one token
//   0x05 setting    a name (word or string) then one value token: "key=value"
//   0x06 line       uint32 length, then the bytes: a record kept as is (a label with two spaces)
// Every value is stored exactly: reading a file gives back the same doubles, bit for bit.

namespace my_torch::binary_format {

// FNV-1a 64 of length bytes from data: the checksum of the binary files (also used by Ensemble).
std::uint64_t checksum(const char *data, std::size_t length);

// True if the bytes start with the signature of a version 2 file.
bool isBinary(const std::string &bytes);

// The records of a text network file (its lines from "input" to the last layer, without the
// "my_torch 1" header and the "end" line) as a complete version 2 file.
std::string encode(const std::string &records);

// The opposite: checks the signature, the version, the length and the checksum, and returns the
// records as text, framed by "my_torch 1" and "end" so that the text reader parses them. Throws
// my_torch::Error, prefixed with path, on a malformed, truncated or corrupted file.
std::string decode(const std::string &bytes, const std::string &path);

} // namespace my_torch::binary_format
