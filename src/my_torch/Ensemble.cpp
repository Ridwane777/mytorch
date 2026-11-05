// Ensembles of networks (#40, ADR-0045).
//
// File format, version 1 (all integers little-endian):
//   bytes 0-7    "MYTORENS"
//   bytes 8-11   format version, uint32 (1)
//   bytes 12-19  payload length in bytes, uint64
//   payload      uint8 combination (0 average, 1 vote, 2 stack), uint32 member count, then per
//                member its weight (IEEE-754 binary64), uint64 length and that many bytes: a
//                complete binary network file (ADR-0042); for stack, uint64 length and the bytes of
//                the combiner network
//   last 8       FNV-1a 64 of the payload, uint64
#include "my_torch/Ensemble.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

#include "BinaryFormat.hpp"
#include "my_torch/Error.hpp"
#include "my_torch/Metrics.hpp"

namespace my_torch {

namespace {

constexpr char signature[8] = {'M', 'Y', 'T', 'O', 'R', 'E', 'N', 'S'};
constexpr std::uint64_t version = 1;
constexpr std::size_t headerSize = 20;

void putLittle(std::string &out, std::uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; i++)
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

// Reads the payload, failing with the file name on any read past its end.
class Cursor {
  public:
    Cursor(const std::string &bytes, std::size_t begin, std::size_t end, const std::string &path)
        : _bytes(bytes), _at(begin), _end(end), _path(path)
    {
    }

    std::uint64_t little(int count)
    {
        need(static_cast<std::size_t>(count));
        std::uint64_t value = 0;
        for (int i = 0; i < count; i++)
            value |= static_cast<std::uint64_t>(static_cast<unsigned char>(_bytes[_at++]))
                     << (8 * i);
        return value;
    }

    std::string block()
    {
        const std::uint64_t length = little(8);
        need(length);
        std::string out = _bytes.substr(_at, length);
        _at += length;
        return out;
    }

    bool done() const noexcept
    {
        return _at == _end;
    }

  private:
    void need(std::uint64_t count) const
    {
        if (count > _end - _at)
            throw Error(_path + ": the ensemble file is truncated");
    }

    const std::string &_bytes;
    std::size_t _at;
    std::size_t _end;
    const std::string &_path;
};

std::string readBytes(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw Error("cannot open '" + path + "'");
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

const char *combinationName(Combination combination) noexcept
{
    switch (combination) {
    case Combination::Vote:
        return "vote";
    case Combination::Stack:
        return "stack";
    default:
        return "average";
    }
}

Combination combinationFromName(const std::string &name)
{
    for (const Combination c : {Combination::Average, Combination::Vote, Combination::Stack})
        if (name == combinationName(c))
            return c;
    throw Error("unknown combination '" + name + "' (expected average, vote or stack)");
}

Ensemble::Ensemble(Combination combination) : _combination(combination)
{
}

void Ensemble::add(Network member, Scalar weight)
{
    if (!std::isfinite(weight) || weight <= 0)
        throw Error("ensemble: a member's weight must be finite and > 0");
    if (!_members.empty() && member.info().labels != labels())
        throw Error("ensemble: every member must have the same labels, in the same order");
    _members.push_back(std::move(member));
    _weights.push_back(weight);
}

void Ensemble::setCombiner(Network combiner)
{
    _combiner.clear();
    _combiner.push_back(std::move(combiner));
}

Combination Ensemble::combination() const noexcept
{
    return _combination;
}

std::size_t Ensemble::size() const noexcept
{
    return _members.size();
}

Network &Ensemble::member(std::size_t index)
{
    return _members.at(index);
}

const Network &Ensemble::member(std::size_t index) const
{
    return _members.at(index);
}

Scalar Ensemble::weight(std::size_t index) const
{
    return _weights.at(index);
}

const std::vector<std::string> &Ensemble::labels() const
{
    if (_members.empty())
        throw Error("ensemble: no member");
    return _members.front().info().labels;
}

void Ensemble::check() const
{
    const std::size_t classes = labels().size();
    if (_combination != Combination::Stack) {
        if (!_combiner.empty())
            throw Error("ensemble: only stack uses a combiner network");
        return;
    }
    if (_combiner.empty())
        throw Error("ensemble: stack needs a combiner network");
    const ModelInfo &info = _combiner.front().info();
    if (info.inputs != _members.size() * classes || info.labels.size() != classes)
        throw Error("ensemble: the combiner must read " +
                    std::to_string(_members.size() * classes) +
                    " inputs (members × labels) and give " + std::to_string(classes) + " outputs");
}

std::vector<Matrix> Ensemble::outputs(const std::vector<Matrix> &inputs)
{
    // A stack's combiner is checked by predict and save, not here: memberOutputs gives the data
    // the combiner is trained on, before it exists.
    if (_combination == Combination::Stack)
        labels();
    else
        check();
    if (inputs.size() != _members.size())
        throw Error("ensemble: " + std::to_string(inputs.size()) + " inputs for " +
                    std::to_string(_members.size()) + " members");
    std::vector<Matrix> out;
    for (std::size_t i = 0; i < _members.size(); i++) {
        if (inputs[i].rows() != inputs.front().rows())
            throw Error("ensemble: the inputs of the members have different row counts");
        out.push_back(_members[i].predict(inputs[i]));
        if (out.back().cols() != labels().size())
            throw Error("ensemble: member " + std::to_string(i + 1) + " gives " +
                        std::to_string(out.back().cols()) + " outputs for " +
                        std::to_string(labels().size()) + " labels");
    }
    return out;
}

Matrix Ensemble::memberOutputs(const std::vector<Matrix> &inputs)
{
    const std::vector<Matrix> out = outputs(inputs);
    const std::size_t classes = labels().size();
    Matrix joined(out.front().rows(), out.size() * classes);
    for (std::size_t r = 0; r < joined.rows(); r++)
        for (std::size_t m = 0; m < out.size(); m++)
            for (std::size_t c = 0; c < classes; c++)
                joined(r, m * classes + c) = out[m](r, c);
    return joined;
}

namespace {

// Weighted mean of the members' outputs.
Matrix weightedMean(const std::vector<Matrix> &out, const std::vector<Scalar> &weights)
{
    Scalar total = 0;
    for (const Scalar w : weights)
        total += w;
    Matrix mean(out.front().rows(), out.front().cols());
    for (std::size_t m = 0; m < out.size(); m++)
        for (std::size_t r = 0; r < mean.rows(); r++)
            for (std::size_t c = 0; c < mean.cols(); c++)
                mean(r, c) += weights[m] / total * out[m](r, c);
    return mean;
}

Matrix voteShares(const std::vector<Matrix> &out, const std::vector<Scalar> &weights)
{
    Scalar total = 0;
    for (const Scalar w : weights)
        total += w;
    Matrix shares(out.front().rows(), out.front().cols());
    for (std::size_t m = 0; m < out.size(); m++) {
        const std::vector<std::size_t> classes = classesOf(out[m]);
        for (std::size_t r = 0; r < shares.rows(); r++)
            shares(r, classes[r]) += weights[m] / total;
    }
    return shares;
}

} // namespace

Matrix Ensemble::predict(const std::vector<Matrix> &inputs)
{
    if (_combination == Combination::Stack) {
        check(); // first: the combiner may be missing
        return _combiner.front().predict(memberOutputs(inputs));
    }
    const std::vector<Matrix> out = outputs(inputs);
    return _combination == Combination::Vote ? voteShares(out, _weights)
                                             : weightedMean(out, _weights);
}

Matrix Ensemble::predict(const Matrix &input)
{
    return predict(std::vector<Matrix>(_members.size(), input));
}

std::vector<std::size_t> Ensemble::classify(const std::vector<Matrix> &inputs)
{
    if (_combination != Combination::Vote)
        return classesOf(predict(inputs));
    const std::vector<Matrix> out = outputs(inputs);
    const Matrix shares = voteShares(out, _weights);
    const Matrix mean = weightedMean(out, _weights);
    std::vector<std::size_t> classes(shares.rows());
    for (std::size_t r = 0; r < shares.rows(); r++) {
        std::size_t best = 0;
        for (std::size_t c = 1; c < shares.cols(); c++) {
            const bool more = shares(r, c) > shares(r, best);
            const bool tie = shares(r, c) == shares(r, best) && mean(r, c) > mean(r, best);
            if (more || tie)
                best = c;
        }
        classes[r] = best;
    }
    return classes;
}

void Ensemble::save(const std::string &path) const
{
    check();
    std::string payload;
    payload.push_back(static_cast<char>(_combination));
    putLittle(payload, _members.size(), 4);
    for (std::size_t i = 0; i < _members.size(); i++) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &_weights[i], sizeof bits);
        putLittle(payload, bits, 8);
        const std::string bytes = binary_format::encode(_members[i].records());
        putLittle(payload, bytes.size(), 8);
        payload += bytes;
    }
    if (_combination == Combination::Stack) {
        const std::string bytes = binary_format::encode(_combiner.front().records());
        putLittle(payload, bytes.size(), 8);
        payload += bytes;
    }
    std::string out(signature, sizeof signature);
    putLittle(out, version, 4);
    putLittle(out, payload.size(), 8);
    out += payload;
    putLittle(out, binary_format::checksum(payload.data(), payload.size()), 8);

    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw Error("cannot write '" + path + "'");
    file << out;
    if (!file)
        throw Error("failed to write '" + path + "'");
}

Ensemble Ensemble::load(const std::string &path)
{
    const std::string bytes = readBytes(path);
    if (bytes.size() < headerSize + 8 || std::memcmp(bytes.data(), signature, 8) != 0)
        throw Error(path + ": not a my_torch ensemble file, or truncated");
    Cursor header(bytes, 8, headerSize, path);
    const std::uint64_t fileVersion = header.little(4);
    if (fileVersion != version)
        throw Error(path + ": ensemble format version " + std::to_string(fileVersion) +
                    " is not supported (this program reads " + std::to_string(version) + ")");
    const std::uint64_t length = header.little(8);
    if (length != bytes.size() - headerSize - 8)
        throw Error(path + ": the ensemble file is truncated or has extra data");
    Cursor sum(bytes, headerSize + length, bytes.size(), path);
    if (sum.little(8) != binary_format::checksum(bytes.data() + headerSize, length))
        throw Error(path + ": checksum mismatch: the file is corrupted");

    Cursor in(bytes, headerSize, headerSize + length, path);
    const std::uint64_t code = in.little(1);
    if (code > static_cast<std::uint64_t>(Combination::Stack))
        throw Error(path + ": unknown combination code " + std::to_string(code));
    Ensemble ensemble(static_cast<Combination>(code));
    const std::uint64_t count = in.little(4);
    if (count == 0)
        throw Error(path + ": an ensemble needs at least one member");
    for (std::uint64_t i = 0; i < count; i++) {
        const std::uint64_t bits = in.little(8);
        Scalar weight = 0;
        std::memcpy(&weight, &bits, sizeof weight);
        const std::string name = path + " (member " + std::to_string(i + 1) + ")";
        try {
            ensemble.add(Network::parse(in.block(), name, false, 0), weight);
        } catch (const Error &error) {
            const std::string message = error.what();
            throw Error(message.rfind(path, 0) == 0 ? message : path + ": " + message);
        }
    }
    if (ensemble._combination == Combination::Stack)
        ensemble.setCombiner(Network::parse(in.block(), path + " (combiner)", false, 0));
    if (!in.done())
        throw Error(path + ": unexpected data after the members");
    try {
        ensemble.check();
    } catch (const Error &error) {
        throw Error(path + ": " + error.what());
    }
    return ensemble;
}

bool Ensemble::isEnsembleFile(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    char start[sizeof signature] = {};
    return in.read(start, sizeof start) && std::memcmp(start, signature, sizeof start) == 0;
}

} // namespace my_torch
