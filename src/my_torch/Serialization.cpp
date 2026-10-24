// Network file, format version 1 (ADR-0012):
//
//   my_torch 1
//   input <encoding> <inputs>
//   labels <count>            then one label per line (labels may contain spaces)
//   loss <type> key=value…
//   optimizer <type> key=value…
//   scheduler <type> key=value…
//   train epochs=… batch=… seed=… shuffle=… patience=… min_delta=… restore_best=…
//   layers <count>            then each layer as written by Layer::save
//   end checksum=<16 hex digits>
//
// The checksum is FNV-1a 64 over every byte before the "end" line. A file edited by hand may end
// with a plain "end": it is then loaded without the check.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

#include "my_torch/Activations.hpp"
#include "my_torch/Dense.hpp"
#include "my_torch/Dropout.hpp"
#include "my_torch/Error.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

namespace {

constexpr int formatVersion = 1;
constexpr const char *magic = "my_torch";

std::uint64_t fnv1a(const std::string &text, std::size_t length)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < length; i++) {
        hash ^= static_cast<unsigned char>(text[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex(std::uint64_t value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; i--, value >>= 4)
        out[static_cast<std::size_t>(i)] = digits[value & 0xF];
    return out;
}

void writeSpec(std::ostream &out, const char *keyword, const ComponentSpec &spec)
{
    out << keyword << ' ' << spec.type;
    for (const auto &[key, value] : spec.params)
        out << ' ' << key << '=' << value;
    out << '\n';
}

// Input width, layer widths and labels must agree; called by save and by load.
void checkConsistency(const ModelInfo &info, const std::vector<std::unique_ptr<Layer>> &layers)
{
    if (info.encoding.empty() || info.encoding.find_first_of(" \t\n") != std::string::npos)
        throw Error("network file: the input encoding must be one non-empty word");
    if (info.inputs == 0)
        throw Error("network file: the input width must be positive");
    if (info.labels.empty())
        throw Error("network file: at least one label is needed");
    for (const std::string &label : info.labels)
        if (label.empty() || label.find('\n') != std::string::npos)
            throw Error("network file: labels must be non-empty single lines");
    std::size_t width = info.inputs;
    for (std::size_t i = 0; i < layers.size(); i++) {
        const auto *dense = dynamic_cast<const Dense *>(layers[i].get());
        if (!dense)
            continue;
        const Matrix &w = dense->weights();
        if (w.rows() != width)
            throw Error("network file: layer " + std::to_string(i + 1) + " expects " +
                        std::to_string(w.rows()) + " inputs but receives " +
                        std::to_string(width));
        width = w.cols();
    }
    if (width != info.labels.size())
        throw Error("network file: the network has " + std::to_string(width) + " outputs for " +
                    std::to_string(info.labels.size()) + " labels");
}

// Line-by-line reader that knows where it is, for error messages.
class Reader {
  public:
    Reader(const std::string &path, std::vector<std::string> lines,
           std::vector<std::size_t> offsets)
        : _path(path), _lines(std::move(lines)), _offsets(std::move(offsets))
    {
    }

    // Byte offset in the file where the line last returned starts.
    std::size_t offset() const noexcept
    {
        return _index == 0 ? 0 : _offsets[_index - 1];
    }

    // Next line, split into words.
    std::vector<std::string> words()
    {
        std::istringstream in(next());
        std::vector<std::string> out;
        for (std::string w; in >> w;)
            out.push_back(w);
        return out;
    }

    std::string next()
    {
        if (_index >= _lines.size())
            fail("unexpected end of file");
        return _lines[_index++];
    }

    // Line starting with keyword; returns the words after it.
    std::vector<std::string> expect(const std::string &keyword)
    {
        auto w = words();
        if (w.empty() || w[0] != keyword)
            fail("expected '" + keyword + "'");
        w.erase(w.begin());
        return w;
    }

    std::vector<Scalar> numbers(std::size_t count)
    {
        const auto w = words();
        if (w.size() != count)
            fail("expected " + std::to_string(count) + " numbers, got " + std::to_string(w.size()));
        std::vector<Scalar> out;
        for (const std::string &text : w)
            out.push_back(number(text));
        return out;
    }

    Scalar number(const std::string &text)
    {
        Scalar value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value))
            fail("'" + text + "' is not a finite number");
        return value;
    }

    std::size_t count(const std::string &text)
    {
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size())
            fail("'" + text + "' is not a non-negative integer");
        return value;
    }

    bool done() const noexcept
    {
        return _index >= _lines.size();
    }

    [[noreturn]] void fail(const std::string &message) const
    {
        throw Error(_path + ":" + std::to_string(_index) + ": " + message);
    }

  private:
    std::string _path;
    std::vector<std::string> _lines;
    std::vector<std::size_t> _offsets;
    std::size_t _index = 0;
};

// "key=value" words, from index first on.
std::vector<std::pair<std::string, std::string>> keyValues(Reader &reader,
                                                           const std::vector<std::string> &w,
                                                           std::size_t first)
{
    std::vector<std::pair<std::string, std::string>> out;
    for (std::size_t i = first; i < w.size(); i++) {
        const std::size_t eq = w[i].find('=');
        if (eq == std::string::npos || eq == 0)
            reader.fail("expected key=value, got '" + w[i] + "'");
        out.emplace_back(w[i].substr(0, eq), w[i].substr(eq + 1));
    }
    return out;
}

// "<type> key=value…" into a spec; the factories check the keys and values.
ComponentSpec readSpec(Reader &reader, const std::vector<std::string> &w)
{
    if (w.empty())
        reader.fail("missing type");
    return {w[0], keyValues(reader, w, 1)};
}

// Settings of a layer line; refuses keys the layer does not take.
std::map<std::string, std::string> layerSettings(Reader &reader, const std::vector<std::string> &w,
                                                 std::size_t first,
                                                 const std::vector<std::string> &allowed)
{
    std::map<std::string, std::string> out;
    for (std::size_t i = first; i < w.size(); i++) {
        const std::size_t eq = w[i].find('=');
        const std::string key = eq == std::string::npos ? w[i] : w[i].substr(0, eq);
        if (eq == std::string::npos ||
            std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            reader.fail(w[0] + ": unexpected '" + w[i] + "'");
        out[key] = w[i].substr(eq + 1);
    }
    return out;
}

std::unique_ptr<Layer> readDense(Reader &reader, const std::vector<std::string> &w)
{
    if (w.size() < 3)
        reader.fail("dense: expected the input and output sizes");
    const std::size_t inputs = reader.count(w[1]);
    const std::size_t outputs = reader.count(w[2]);
    if (inputs == 0 || outputs == 0)
        reader.fail("dense: sizes must be positive");
    auto settings = layerSettings(reader, w, 3, {"l1", "l2"});
    Regularization reg;
    if (settings.count("l1"))
        reg.l1 = reader.number(settings["l1"]);
    if (settings.count("l2"))
        reg.l2 = reader.number(settings["l2"]);
    auto dense = std::make_unique<Dense>(inputs, outputs, 0, reg);
    for (std::size_t r = 0; r < inputs; r++) {
        const auto row = reader.numbers(outputs);
        for (std::size_t c = 0; c < outputs; c++)
            dense->weights()(r, c) = row[c];
    }
    const auto bias = reader.numbers(outputs);
    for (std::size_t c = 0; c < outputs; c++)
        dense->bias()(0, c) = bias[c];
    return dense;
}

template <typename T> std::unique_ptr<Layer> readPlain(Reader &reader,
                                                       const std::vector<std::string> &w)
{
    layerSettings(reader, w, 1, {});
    return std::make_unique<T>();
}

std::unique_ptr<Layer> readLeakyReLU(Reader &reader, const std::vector<std::string> &w)
{
    auto settings = layerSettings(reader, w, 1, {"slope"});
    return std::make_unique<LeakyReLU>(settings.count("slope") ? reader.number(settings["slope"])
                                                               : 0.01);
}

std::unique_ptr<Layer> readDropout(Reader &reader, const std::vector<std::string> &w)
{
    auto settings = layerSettings(reader, w, 1, {"rate", "seed"});
    if (!settings.count("rate"))
        reader.fail("dropout: missing rate");
    std::uint64_t seed = 42;
    if (settings.count("seed")) {
        const std::string &text = settings["seed"];
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
        if (error != std::errc{} || end != text.data() + text.size())
            reader.fail("dropout: bad seed '" + text + "'");
    }
    return std::make_unique<Dropout>(reader.number(settings["rate"]), seed);
}

// One entry per layer type: adding a layer to the format is one line here plus its save().
using LayerReader =
    std::function<std::unique_ptr<Layer>(Reader &, const std::vector<std::string> &)>;
const std::map<std::string, LayerReader> &layerReaders()
{
    static const std::map<std::string, LayerReader> readers = {
        {"dense", readDense},
        {"relu", readPlain<ReLU>},
        {"leaky_relu", readLeakyReLU},
        {"sigmoid", readPlain<Sigmoid>},
        {"tanh", readPlain<Tanh>},
        {"softmax", readPlain<Softmax>},
        {"dropout", readDropout},
    };
    return readers;
}

void writeConfig(std::ostream &out, const TrainConfig &c)
{
    out << "train epochs=" << c.epochs << " batch=" << c.batchSize << " seed=" << c.seed
        << " shuffle=" << c.shuffle << " patience=" << c.patience << " min_delta=" << c.minDelta
        << " restore_best=" << c.restoreBest << '\n';
}

TrainConfig readConfig(Reader &reader)
{
    TrainConfig c;
    for (const auto &[key, value] : keyValues(reader, reader.expect("train"), 0)) {
        const auto flag = [&] {
            if (value != "0" && value != "1")
                reader.fail("train: " + key + " must be 0 or 1");
            return value == "1";
        };
        if (key == "epochs")
            c.epochs = reader.count(value);
        else if (key == "batch")
            c.batchSize = reader.count(value);
        else if (key == "seed")
            c.seed = reader.count(value);
        else if (key == "shuffle")
            c.shuffle = flag();
        else if (key == "patience")
            c.patience = reader.count(value);
        else if (key == "min_delta")
            c.minDelta = reader.number(value);
        else if (key == "restore_best")
            c.restoreBest = flag();
        else
            reader.fail("train: unknown setting '" + key + "'");
    }
    return c;
}

} // namespace

ModelInfo &Network::info() noexcept
{
    return _info;
}

const ModelInfo &Network::info() const noexcept
{
    return _info;
}

void Network::save(const std::string &path) const
{
    checkConsistency(_info, _layers);
    std::ostringstream body;
    body.precision(17); // shortest precision that reads back every double exactly
    body << magic << ' ' << formatVersion << '\n';
    body << "input " << _info.encoding << ' ' << _info.inputs << '\n';
    body << "labels " << _info.labels.size() << '\n';
    for (const std::string &label : _info.labels)
        body << label << '\n';
    writeSpec(body, "loss", _info.training.loss);
    writeSpec(body, "optimizer", _info.training.optimizer);
    writeSpec(body, "scheduler", _info.training.scheduler);
    writeConfig(body, _info.training.config);
    body << "layers " << _layers.size() << '\n';
    for (const auto &layer : _layers)
        layer->save(body);
    const std::string text = body.str();

    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw Error("cannot write '" + path + "'");
    out << text << "end checksum=" << hex(fnv1a(text, text.size())) << '\n';
    if (!out)
        throw Error("failed to write '" + path + "'");
}

Network Network::load(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw Error("cannot open '" + path + "'");
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    // Split into lines, remembering where each starts: the checksum covers what precedes "end".
    std::vector<std::string> lines;
    std::vector<std::size_t> offsets;
    for (std::size_t start = 0; start < text.size();) {
        std::size_t end = text.find('\n', start);
        if (end == std::string::npos)
            end = text.size();
        std::string line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        offsets.push_back(start);
        lines.push_back(line);
        start = end + 1;
    }
    Reader reader(path, lines, offsets);

    const auto header = reader.words();
    if (header.size() != 2 || header[0] != magic)
        reader.fail("not a my_torch network file");
    if (header[1] != std::to_string(formatVersion))
        reader.fail("format version " + header[1] + " is not supported (this program reads " +
                    std::to_string(formatVersion) + ")");

    Network net;
    ModelInfo &info = net._info;
    const auto input = reader.expect("input");
    if (input.size() != 2)
        reader.fail("input: expected an encoding name and a width");
    info.encoding = input[0];
    info.inputs = reader.count(input[1]);
    const auto labels = reader.expect("labels");
    if (labels.size() != 1)
        reader.fail("labels: expected a count");
    const std::size_t labelCount = reader.count(labels[0]);
    for (std::size_t i = 0; i < labelCount; i++)
        info.labels.push_back(reader.next());
    info.training.loss = readSpec(reader, reader.expect("loss"));
    info.training.optimizer = readSpec(reader, reader.expect("optimizer"));
    info.training.scheduler = readSpec(reader, reader.expect("scheduler"));
    info.training.config = readConfig(reader);

    const auto layers = reader.expect("layers");
    if (layers.size() != 1)
        reader.fail("layers: expected a count");
    const std::size_t layerCount = reader.count(layers[0]);
    for (std::size_t i = 0; i < layerCount; i++) {
        const auto w = reader.words();
        if (w.empty())
            reader.fail("expected a layer");
        const auto found = layerReaders().find(w[0]);
        if (found == layerReaders().end())
            reader.fail("unknown layer type '" + w[0] + "'");
        try {
            net.add(found->second(reader, w));
        } catch (const Error &error) {
            const std::string message = error.what();
            if (message.rfind(path + ":", 0) == 0)
                throw;
            reader.fail(message);
        }
    }

    const auto end = reader.words();
    if (end.empty() || end[0] != "end")
        reader.fail("expected 'end'");
    const std::size_t endOffset = reader.offset();
    const bool checked = end.size() == 2 && end[1].rfind("checksum=", 0) == 0;
    if (!checked && end.size() != 1)
        reader.fail("unexpected data after 'end'");
    while (!reader.done())
        if (!reader.next().empty())
            reader.fail("unexpected data after 'end'");
    if (checked && end[1].substr(9) != hex(fnv1a(text, endOffset)))
        reader.fail("checksum mismatch: the file is corrupted");
    try {
        checkConsistency(info, net._layers);
    } catch (const Error &error) {
        throw Error(path + ": " + error.what());
    }
    return net;
}

} // namespace my_torch
