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
#include "my_torch/Factory.hpp"
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
// Line-by-line reader that knows where it is, for error messages.
class Reader {
  public:
    struct Line {
        std::string text;
        std::size_t offset; // byte where the line starts in the file
        std::size_t number; // 1-based line number in the file
    };

    Reader(const std::string &path, std::vector<Line> lines)
        : _path(path), _lines(std::move(lines))
    {
    }

    // Byte offset in the file where the line last returned starts.
    std::size_t offset() const noexcept
    {
        return _index == 0 ? 0 : _lines[_index - 1].offset;
    }

    // Next line, split into words.
    std::vector<std::string> words()
    {
        return split(next());
    }

    std::string next()
    {
        if (_index >= _lines.size())
            fail("unexpected end of file");
        return _lines[_index++].text;
    }

    // First word of the next line without consuming it ("" at the end).
    std::string peek() const
    {
        if (_index >= _lines.size())
            return "";
        const auto w = split(_lines[_index].text);
        return w.empty() ? "" : w[0];
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

    std::uint64_t count(const std::string &text)
    {
        std::uint64_t value = 0;
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
        const std::size_t line = _index == 0 ? 0 : _lines[_index - 1].number;
        throw Error(_path + ":" + std::to_string(line) + ": " + message);
    }

  private:
    static std::vector<std::string> split(const std::string &line)
    {
        std::istringstream in(line);
        std::vector<std::string> out;
        for (std::string w; in >> w;)
            out.push_back(w);
        return out;
    }

    std::string _path;
    std::vector<Line> _lines;
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

// SplitMix64: a fully specified mix, so derived seeds are the same on every toolchain.
std::uint64_t mix(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// What a layer reader needs besides its line.
struct Context {
    bool config;        // a configuration file: no weights, weights drawn from seeds
    std::uint64_t seed; // seed of this network (config only)
    std::size_t layer;  // index of the layer being read

    std::uint64_t layerSeed() const
    {
        return mix(mix(seed) ^ static_cast<std::uint64_t>(layer));
    }
};

std::unique_ptr<Layer> readDense(Reader &reader, const std::vector<std::string> &w,
                                 const Context &ctx)
{
    if (w.size() < 3)
        reader.fail("dense: expected the input and output sizes");
    const std::size_t inputs = reader.count(w[1]);
    const std::size_t outputs = reader.count(w[2]);
    if (inputs == 0 || outputs == 0)
        reader.fail("dense: sizes must be positive");
    auto settings = layerSettings(reader, w, 3,
                                  ctx.config ? std::vector<std::string>{"l1", "l2", "init"}
                                             : std::vector<std::string>{"l1", "l2"});
    Regularization reg;
    if (settings.count("l1"))
        reg.l1 = reader.number(settings["l1"]);
    if (settings.count("l2"))
        reg.l2 = reader.number(settings["l2"]);
    // Only the provisional uniform initialization exists until #6; the key is reserved for it.
    if (settings.count("init") && settings["init"] != "uniform")
        reader.fail("dense: unknown init '" + settings["init"] + "' (available: uniform)");
    auto dense = std::make_unique<Dense>(inputs, outputs,
                                         static_cast<std::uint32_t>(ctx.layerSeed()), reg);
    if (ctx.config)
        return dense;
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

template <typename T>
std::unique_ptr<Layer> readPlain(Reader &reader, const std::vector<std::string> &w,
                                 const Context &)
{
    layerSettings(reader, w, 1, {});
    return std::make_unique<T>();
}

std::unique_ptr<Layer> readLeakyReLU(Reader &reader, const std::vector<std::string> &w,
                                     const Context &)
{
    auto settings = layerSettings(reader, w, 1, {"slope"});
    return std::make_unique<LeakyReLU>(settings.count("slope") ? reader.number(settings["slope"])
                                                               : 0.01);
}

std::unique_ptr<Layer> readDropout(Reader &reader, const std::vector<std::string> &w,
                                   const Context &ctx)
{
    auto settings = layerSettings(reader, w, 1, {"rate", "seed"});
    if (!settings.count("rate"))
        reader.fail("dropout: missing rate");
    // A configuration without a seed gives each generated network its own mask sequence.
    const std::uint64_t seed = settings.count("seed") ? reader.count(settings["seed"])
                               : ctx.config           ? ctx.layerSeed()
                                                      : 42;
    return std::make_unique<Dropout>(reader.number(settings["rate"]), seed);
}

// One entry per layer type: adding a layer to the format is one line here plus its save().
using LayerReader = std::function<std::unique_ptr<Layer>(
    Reader &, const std::vector<std::string> &, const Context &)>;
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

TrainConfig readTrainConfig(Reader &reader, const std::vector<std::string> &w)
{
    TrainConfig c;
    for (const auto &[key, value] : keyValues(reader, w, 0)) {
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

std::string readFile(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw Error("cannot open '" + path + "'");
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// Splits into lines with their offset and number. A configuration also drops blank lines and
// comments (lines whose first non-blank character is '#').
std::vector<Reader::Line> splitLines(const std::string &text, bool config)
{
    std::vector<Reader::Line> lines;
    std::size_t number = 0;
    for (std::size_t start = 0; start < text.size();) {
        std::size_t end = text.find('\n', start);
        if (end == std::string::npos)
            end = text.size();
        std::string line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        number++;
        const std::size_t first = line.find_first_not_of(" \t");
        if (!config || (first != std::string::npos && line[first] != '#'))
            lines.push_back({line, start, number});
        start = end + 1;
    }
    return lines;
}

// Builds the training objects once, so that an unknown loss, optimizer or scheduler is reported
// when the file is read rather than when training starts.
void checkTraining(const std::string &path, const TrainingSpec &training)
{
    try {
        makeLoss(training.loss);
        makeOptimizer(training.optimizer);
        makeScheduler(training.scheduler);
    } catch (const Error &error) {
        throw Error(path + ": " + error.what());
    }
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

Network Network::parse(const std::string &path, bool config, std::uint64_t index)
{
    const std::string text = readFile(path);
    Reader reader(path, splitLines(text, config));
    Network net;
    ModelInfo &info = net._info;

    if (!config) {
        const auto header = reader.words();
        if (header.size() != 2 || header[0] != magic)
            reader.fail("not a my_torch network file");
        if (header[1] != std::to_string(formatVersion))
            reader.fail("format version " + header[1] + " is not supported (this program reads " +
                        std::to_string(formatVersion) + ")");
    }
    const auto input = reader.expect("input");
    if (input.size() != 2)
        reader.fail("input: expected an encoding name and a width");
    info.encoding = input[0];
    info.inputs = reader.count(input[1]);
    const auto labels = reader.expect("labels");
    if (labels.size() != 1)
        reader.fail("labels: expected a count");
    const std::uint64_t labelCount = reader.count(labels[0]);
    for (std::uint64_t i = 0; i < labelCount; i++)
        info.labels.push_back(reader.next());

    // A network file has the four sections in this order. A configuration may give them in any
    // order or omit them (the defaults apply), and adds "init seed=…".
    std::uint64_t initSeed = 1;
    const auto readSection = [&](const std::string &keyword) {
        auto w = reader.expect(keyword);
        if (keyword == "train")
            info.training.config = readTrainConfig(reader, w);
        else if (keyword == "loss")
            info.training.loss = readSpec(reader, w);
        else if (keyword == "optimizer")
            info.training.optimizer = readSpec(reader, w);
        else
            info.training.scheduler = readSpec(reader, w);
    };
    if (!config) {
        for (const char *keyword : {"loss", "optimizer", "scheduler", "train"})
            readSection(keyword);
    } else {
        std::vector<std::string> seen;
        for (std::string keyword = reader.peek(); keyword != "layers"; keyword = reader.peek()) {
            if (std::find(seen.begin(), seen.end(), keyword) != seen.end()) {
                reader.words();
                reader.fail("'" + keyword + "' given twice");
            }
            seen.push_back(keyword);
            if (keyword == "init") {
                for (const auto &[key, value] : keyValues(reader, reader.expect("init"), 0)) {
                    if (key != "seed")
                        reader.fail("init: unknown setting '" + key + "'");
                    initSeed = reader.count(value);
                }
            } else if (keyword == "loss" || keyword == "optimizer" || keyword == "scheduler" ||
                       keyword == "train") {
                readSection(keyword);
            } else {
                reader.words();
                reader.fail(keyword.empty() ? "expected 'layers'"
                                            : "unknown keyword '" + keyword + "'");
            }
        }
    }

    const auto layers = reader.expect("layers");
    if (layers.size() != 1)
        reader.fail("layers: expected a count");
    const std::uint64_t layerCount = reader.count(layers[0]);
    for (std::uint64_t i = 0; i < layerCount; i++) {
        const auto w = reader.words();
        if (w.empty())
            reader.fail("expected a layer");
        const auto found = layerReaders().find(w[0]);
        if (found == layerReaders().end())
            reader.fail("unknown layer type '" + w[0] + "'");
        const Context ctx{config, initSeed + index, static_cast<std::size_t>(i)};
        try {
            net.add(found->second(reader, w, ctx));
        } catch (const Error &error) {
            const std::string message = error.what();
            if (message.rfind(path + ":", 0) == 0)
                throw;
            reader.fail(message);
        }
    }

    if (config) {
        if (!reader.done()) {
            reader.words();
            reader.fail("unexpected data after the layers");
        }
    } else {
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
    }
    try {
        checkConsistency(info, net._layers);
    } catch (const Error &error) {
        throw Error(path + ": " + error.what());
    }
    checkTraining(path, info.training);
    return net;
}

Network Network::load(const std::string &path)
{
    return parse(path, false, 0);
}

Network Network::fromConfig(const std::string &path, std::uint64_t index)
{
    return parse(path, true, index);
}

} // namespace my_torch
