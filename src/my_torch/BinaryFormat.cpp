#include "BinaryFormat.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <vector>

#include "my_torch/Error.hpp"

namespace my_torch::binary_format {

namespace {

constexpr char signature[8] = {'M', 'Y', 'T', 'O', 'R', 'C', 'H', '\0'};
constexpr std::uint32_t version = 2;
constexpr std::size_t headerSize = 8 + 4 + 8;

enum Tag : unsigned char {
    EndOfRecord = 0x00,
    Word = 0x01,
    String = 0x02,
    Integer = 0x03,
    Reals = 0x04,
    Setting = 0x05,
    Line = 0x06,
};

// The keywords of the library's records. Append only: a word's index is its code in every file
// already written. Names the library does not define (encodings, labels) are strings.
constexpr std::array words = {
    "input", "labels", "loss", "optimizer", "scheduler", "train", "layers",
    // layers
    "dense", "relu", "leaky_relu", "sigmoid", "tanh", "softmax", "dropout", "batch_norm",
    "layer_norm", "conv2d", "max_pool2d", "avg_pool2d", "global_avg_pool", "to_planes",
    "board_planes",
    // losses, optimizers, schedulers
    "cross_entropy", "binary_cross_entropy", "mse", "mae", "huber", "sgd", "adam", "adamw",
    "rmsprop", "adagrad", "none", "step", "exponential", "cosine",
    // initialisations
    "he_normal", "he_uniform", "xavier_normal", "xavier_uniform", "lecun_normal", "lecun_uniform",
    "orthogonal", "uniform", "constant",
    // setting names
    "epochs", "batch", "seed", "shuffle", "patience", "min_delta", "restore_best", "validation",
    "validation_tail", "clip_norm", "clip_value", "lr", "momentum", "nesterov", "beta1", "beta2",
    "epsilon", "weight_decay", "amsgrad", "decay", "gamma", "base", "min", "warmup", "delta",
    "smoothing", "rate", "slope", "l1", "l2", "init", "channels", "height", "width", "filters",
    "kernel", "stride", "padding", "size", "inputs", "scalars", "count"};
static_assert(words.size() < 256, "a word code is one byte");

const std::map<std::string, unsigned char> &wordCodes()
{
    static const std::map<std::string, unsigned char> codes = [] {
        std::map<std::string, unsigned char> m;
        for (std::size_t i = 0; i < words.size(); i++)
            m.emplace(words[i], static_cast<unsigned char>(i));
        return m;
    }();
    return codes;
}

// ---- writing -----------------------------------------------------------------------------------

void putLittle(std::string &out, std::uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; i++, value >>= 8)
        out.push_back(static_cast<char>(value & 0xFF));
}

void putLength(std::string &out, std::size_t length)
{
    if (length > UINT32_MAX)
        throw Error("network file: a record is too long for the binary format");
    putLittle(out, length, 4);
}

void putBytes(std::string &out, const std::string &text)
{
    putLength(out, text.size());
    out += text;
}

// The text of a double as Network::save writes it (an ostream at precision 17).
std::string formatReal(double value)
{
    char buffer[40];
    const int n = std::snprintf(buffer, sizeof buffer, "%.17g", value);
    return std::string(buffer, static_cast<std::size_t>(n));
}

bool isInteger(const std::string &token)
{
    if (token.empty() || token.size() > 19 || (token.size() > 1 && token[0] == '0'))
        return false;
    for (char c : token)
        if (c < '0' || c > '9')
            return false;
    return true;
}

// The value of a token stored as a binary64, if writing it back gives the same text.
std::optional<double> asReal(const std::string &token)
{
    double value = 0;
    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc() || end != token.data() + token.size() || formatReal(value) != token)
        return std::nullopt;
    return value;
}

// One token that is not part of a run of reals: an integer, a word, or a string.
void putName(std::string &out, const std::string &token)
{
    const auto found = wordCodes().find(token);
    if (found != wordCodes().end()) {
        out.push_back(static_cast<char>(Word));
        out.push_back(static_cast<char>(found->second));
    } else {
        out.push_back(static_cast<char>(String));
        putBytes(out, token);
    }
}

void putValue(std::string &out, const std::string &token)
{
    if (isInteger(token)) {
        out.push_back(static_cast<char>(Integer));
        for (std::uint64_t v = std::stoull(token);; v >>= 7) {
            const auto byte = static_cast<unsigned char>(v & 0x7F);
            if (v < 0x80) {
                out.push_back(static_cast<char>(byte));
                break;
            }
            out.push_back(static_cast<char>(byte | 0x80));
        }
    } else if (const auto real = asReal(token)) {
        out.push_back(static_cast<char>(Reals));
        putLength(out, 1);
        putLittle(out, std::bit_cast<std::uint64_t>(*real), 8);
    } else {
        putName(out, token);
    }
}

std::vector<std::string> splitSpaces(const std::string &line)
{
    std::vector<std::string> tokens;
    std::size_t start = 0;
    for (std::size_t space = line.find(' '); space != std::string::npos;
         space = line.find(' ', start)) {
        tokens.push_back(line.substr(start, space - start));
        start = space + 1;
    }
    tokens.push_back(line.substr(start));
    return tokens;
}

void putRecord(std::string &out, const std::string &line)
{
    const std::vector<std::string> tokens = splitSpaces(line);
    for (const std::string &token : tokens)
        if (token.empty() || token.find_first_of("\t\r\n") != std::string::npos) {
            // Two spaces in a row, a leading or trailing space, a tab: kept byte for byte.
            out.push_back(static_cast<char>(Line));
            putBytes(out, line);
            return;
        }
    for (std::size_t i = 0; i < tokens.size();) {
        std::vector<double> run; // consecutive reals, converted once
        for (std::size_t j = i; j < tokens.size() && !isInteger(tokens[j]); j++) {
            const auto real = asReal(tokens[j]);
            if (!real)
                break;
            run.push_back(*real);
        }
        if (!run.empty()) {
            out.push_back(static_cast<char>(Reals));
            putLength(out, run.size());
            for (double value : run)
                putLittle(out, std::bit_cast<std::uint64_t>(value), 8);
            i += run.size();
            continue;
        }
        const std::size_t equal = tokens[i].find('=');
        if (equal != std::string::npos && equal > 0) {
            out.push_back(static_cast<char>(Setting));
            putName(out, tokens[i].substr(0, equal));
            putValue(out, tokens[i].substr(equal + 1));
        } else {
            putValue(out, tokens[i]);
        }
        i++;
    }
    out.push_back(static_cast<char>(EndOfRecord));
}

std::uint64_t fnv1a(const char *data, std::size_t length)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < length; i++) {
        hash ^= static_cast<unsigned char>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ---- reading -----------------------------------------------------------------------------------

class Cursor {
  public:
    Cursor(const std::string &bytes, std::size_t begin, std::size_t end, const std::string &path)
        : _bytes(bytes), _at(begin), _end(end), _path(path)
    {
    }

    bool done() const noexcept
    {
        return _at == _end;
    }

    [[noreturn]] void fail(const std::string &what) const
    {
        throw Error(_path + ": corrupted network file (" + what + " at byte " +
                    std::to_string(_at) + ")");
    }

    unsigned char byte()
    {
        if (_at >= _end)
            fail("unexpected end of the data");
        return static_cast<unsigned char>(_bytes[_at++]);
    }

    std::uint64_t little(int count)
    {
        std::uint64_t value = 0;
        for (int i = 0; i < count; i++)
            value |= static_cast<std::uint64_t>(byte()) << (8 * i);
        return value;
    }

    std::string bytes()
    {
        const std::uint64_t length = little(4);
        if (length > _end - _at)
            fail("a length beyond the end of the data");
        std::string text = _bytes.substr(_at, length);
        _at += length;
        return text;
    }

  private:
    const std::string &_bytes;
    std::size_t _at;
    std::size_t _end;
    const std::string &_path;
};

std::string readName(Cursor &in, unsigned char tag)
{
    if (tag == Word) {
        const unsigned char code = in.byte();
        if (code >= words.size())
            in.fail("unknown word code " + std::to_string(code));
        return words[code];
    }
    if (tag == String)
        return in.bytes();
    in.fail("expected a name");
}

// One token; Reals appends its values separated by spaces.
std::string readValue(Cursor &in, unsigned char tag)
{
    if (tag == Integer) {
        std::uint64_t value = 0;
        for (int shift = 0;; shift += 7) {
            if (shift > 63)
                in.fail("an integer longer than 64 bits");
            const unsigned char b = in.byte();
            value |= static_cast<std::uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80))
                break;
        }
        return std::to_string(value);
    }
    if (tag == Reals) {
        const std::uint64_t count = in.little(4);
        if (count == 0)
            in.fail("an empty run of reals");
        std::string text;
        for (std::uint64_t i = 0; i < count; i++)
            text += (i ? " " : "") + formatReal(std::bit_cast<double>(in.little(8)));
        return text;
    }
    return readName(in, tag);
}

} // namespace

bool isBinary(const std::string &bytes)
{
    return bytes.size() >= sizeof signature && std::memcmp(bytes.data(), signature, 8) == 0;
}

std::string encode(const std::string &records)
{
    std::string payload;
    payload.reserve(records.size() / 2);
    std::size_t start = 0;
    while (start < records.size()) {
        std::size_t end = records.find('\n', start);
        if (end == std::string::npos)
            end = records.size();
        putRecord(payload, records.substr(start, end - start));
        start = end + 1;
    }
    std::string out(signature, sizeof signature);
    putLittle(out, version, 4);
    putLittle(out, payload.size(), 8);
    out += payload;
    putLittle(out, fnv1a(payload.data(), payload.size()), 8);
    return out;
}

std::string decode(const std::string &bytes, const std::string &path)
{
    if (!isBinary(bytes) || bytes.size() < headerSize + 8)
        throw Error(path + ": not a binary my_torch network file, or truncated");
    Cursor header(bytes, 8, headerSize, path);
    const std::uint64_t fileVersion = header.little(4);
    if (fileVersion != version)
        throw Error(path + ": binary format version " + std::to_string(fileVersion) +
                    " is not supported (this program reads " + std::to_string(version) + ")");
    const std::uint64_t length = header.little(8);
    if (length != bytes.size() - headerSize - 8)
        throw Error(path + ": the network file is truncated or has extra data (" +
                    std::to_string(bytes.size()) + " bytes, " +
                    std::to_string(length + headerSize + 8) + " expected)");
    Cursor sum(bytes, headerSize + length, bytes.size(), path);
    if (sum.little(8) != fnv1a(bytes.data() + headerSize, length))
        throw Error(path + ": checksum mismatch: the file is corrupted");

    std::string text = "my_torch 1\n";
    Cursor in(bytes, headerSize, headerSize + length, path);
    while (!in.done()) {
        bool first = true;
        for (unsigned char tag = in.byte(); tag != EndOfRecord; tag = in.byte()) {
            if (tag == Line) {
                text += in.bytes();
                break;
            }
            if (!first)
                text += ' ';
            first = false;
            if (tag == Setting) {
                text += readName(in, in.byte());
                text += '=';
                text += readValue(in, in.byte());
            } else if (tag == Word || tag == String || tag == Integer || tag == Reals) {
                text += readValue(in, tag);
            } else {
                in.fail("unknown token type " + std::to_string(tag));
            }
        }
        text += '\n';
    }
    return text + "end\n";
}

} // namespace my_torch::binary_format
