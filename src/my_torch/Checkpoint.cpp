#include "my_torch/Checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>

#include "my_torch/Error.hpp"

// Checkpoint file, format version 1 (ADR-0032):
//
//   my_torch_checkpoint 1
//   fingerprint <16 hex digits>
//   epoch <finished epochs>
//   shuffle <length>         then that many characters: the generator's text state
//   history <count>          then per epoch: trainLoss validationLoss penalty trainAccuracy
//                            validationAccuracy
//   early <best> <sinceBest>
//   best_weights             then matrices
//   weights                  then matrices
//   optimizer <length>       then that many characters
//   layers <count>           then per layer: <length> and that many characters
//   end checksum=<16 hex digits>
//
// Matrices: "<count>", then per matrix "<rows> <cols>" and its values in row-major order.
// The checksum is FNV-1a 64 over every byte before the "end" line.

namespace my_torch {

namespace {

constexpr const char *magic = "my_torch_checkpoint";
constexpr int formatVersion = 1;

std::string hex(std::uint64_t value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; i--, value >>= 4)
        out[static_cast<std::size_t>(i)] = digits[value & 0xF];
    return out;
}

[[noreturn]] void corrupted(const std::string &path, const std::string &what)
{
    throw Error("checkpoint '" + path + "': " + what);
}

void expect(std::istream &in, const std::string &path, const char *keyword)
{
    std::string word;
    if (!(in >> word) || word != keyword)
        corrupted(path, std::string("expected '") + keyword + "'");
}

void writeBlock(std::ostream &out, const std::string &block)
{
    out << block.size() << '\n' << block << '\n';
}

std::string readBlock(std::istream &in, const std::string &path)
{
    std::size_t length = 0;
    if (!(in >> length) || in.get() != '\n')
        corrupted(path, "malformed block length");
    std::string block(length, '\0');
    if (!in.read(block.data(), static_cast<std::streamsize>(length)))
        corrupted(path, "truncated block");
    return block;
}

} // namespace

void writeMatrices(std::ostream &out, const std::vector<Matrix> &matrices)
{
    const auto precision = out.precision(17);
    out << matrices.size() << '\n';
    for (const Matrix &m : matrices) {
        out << m.rows() << ' ' << m.cols() << '\n';
        for (std::size_t i = 0; i < m.size(); i++)
            out << m.data()[i] << (i + 1 < m.size() ? ' ' : '\n');
    }
    out.precision(precision);
}

std::vector<Matrix> readMatrices(std::istream &in)
{
    std::size_t count = 0;
    if (!(in >> count))
        throw Error("state: expected a matrix count");
    std::vector<Matrix> matrices;
    for (std::size_t k = 0; k < count; k++) {
        std::size_t rows = 0;
        std::size_t cols = 0;
        if (!(in >> rows >> cols))
            throw Error("state: expected the shape of a matrix");
        Matrix m(rows, cols);
        for (std::size_t i = 0; i < m.size(); i++)
            if (!(in >> m.data()[i]))
                throw Error("state: expected " + std::to_string(m.size()) + " values");
        matrices.push_back(std::move(m));
    }
    return matrices;
}

std::uint64_t fingerprintOf(const std::string &text)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void saveCheckpoint(const std::string &path, std::uint64_t fingerprint, const TrainingState &state)
{
    std::ostringstream body;
    body.precision(17);
    body << magic << ' ' << formatVersion << '\n';
    body << "fingerprint " << hex(fingerprint) << '\n';
    body << "epoch " << state.epoch << '\n';
    body << "shuffle ";
    writeBlock(body, state.shuffle);
    body << "history " << state.history.size() << '\n';
    for (const EpochStats &s : state.history)
        body << s.trainLoss << ' ' << s.validationLoss << ' ' << s.penalty << ' '
             << s.trainAccuracy << ' ' << s.validationAccuracy << '\n';
    body << "early " << state.best << ' ' << state.sinceBest << '\n';
    body << "best_weights\n";
    writeMatrices(body, state.bestWeights);
    body << "weights\n";
    writeMatrices(body, state.weights);
    body << "optimizer ";
    writeBlock(body, state.optimizer);
    body << "layers " << state.layers.size() << '\n';
    for (const std::string &layer : state.layers)
        writeBlock(body, layer);
    const std::string text = body.str();

    const std::string temporary = path + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        if (!out)
            throw Error("cannot write checkpoint '" + temporary + "'");
        out << text << "end checksum=" << hex(fingerprintOf(text)) << '\n';
        out.flush();
        if (!out)
            throw Error("failed to write checkpoint '" + temporary + "'");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
        throw Error("cannot replace checkpoint '" + path + "': " + error.message());
}

std::optional<TrainingState> loadCheckpoint(const std::string &path, std::uint64_t fingerprint)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (!std::filesystem::exists(path))
            return std::nullopt;
        throw Error("cannot read checkpoint '" + path + "'");
    }
    std::ostringstream raw;
    raw << file.rdbuf();
    const std::string text = raw.str();
    const std::size_t end = text.rfind("end checksum=");
    if (end == std::string::npos || (end > 0 && text[end - 1] != '\n'))
        corrupted(path, "no end line, the file is truncated");
    const std::string body = text.substr(0, end);
    std::string stored = text.substr(end + 13);
    while (!stored.empty() && (stored.back() == '\n' || stored.back() == '\r'))
        stored.pop_back();
    if (stored != hex(fingerprintOf(body)))
        corrupted(path, "checksum mismatch, the file is corrupted");

    std::istringstream in(body);
    std::string word;
    int version = 0;
    if (!(in >> word >> version) || word != magic)
        corrupted(path, "not a my_torch checkpoint");
    if (version != formatVersion)
        corrupted(path, "format version " + std::to_string(version) + " is not supported");
    expect(in, path, "fingerprint");
    if (!(in >> word) || word != hex(fingerprint))
        throw Error("checkpoint '" + path +
                    "' was written by another training (other settings, layers or data): delete "
                    "it to start again");

    TrainingState state;
    expect(in, path, "epoch");
    if (!(in >> state.epoch))
        corrupted(path, "malformed epoch");
    expect(in, path, "shuffle");
    state.shuffle = readBlock(in, path);
    expect(in, path, "history");
    std::size_t epochs = 0;
    if (!(in >> epochs))
        corrupted(path, "malformed history");
    for (std::size_t e = 0; e < epochs; e++) {
        EpochStats s;
        if (!(in >> s.trainLoss >> s.validationLoss >> s.penalty >> s.trainAccuracy >>
              s.validationAccuracy))
            corrupted(path, "malformed history");
        state.history.push_back(s);
    }
    expect(in, path, "early");
    if (!(in >> state.best >> state.sinceBest))
        corrupted(path, "malformed early stopping state");
    try {
        expect(in, path, "best_weights");
        state.bestWeights = readMatrices(in);
        expect(in, path, "weights");
        state.weights = readMatrices(in);
    } catch (const Error &e) {
        corrupted(path, e.what());
    }
    expect(in, path, "optimizer");
    state.optimizer = readBlock(in, path);
    expect(in, path, "layers");
    std::size_t layers = 0;
    if (!(in >> layers))
        corrupted(path, "malformed layer count");
    for (std::size_t i = 0; i < layers; i++)
        state.layers.push_back(readBlock(in, path));
    return state;
}

} // namespace my_torch
