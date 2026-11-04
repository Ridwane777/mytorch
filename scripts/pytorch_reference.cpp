// The my_torch side of the PyTorch reference (#42, ADR-0040,
// docs/performance/22-pytorch-reference.md).
// Three commands, each writing plain binary files that comparison/*.py reads (numpy, no
// serialization format of the other library): they let PyTorch start from exactly our weights, see
// exactly our batches, and be compared with exactly our gradients. Only the my_torch headers and
// the analyzer's position reader are used here; nothing of PyTorch is.
//
//   pytorch_reference export CONFIG DATA OUT_PREFIX
//       encodes every position of DATA with the encoding named by CONFIG and writes
//       OUT_PREFIX.x.u8 (rows × inputs bytes, each 0 or 1: checked) and OUT_PREFIX.y.u8 (the class
//       of each row, one byte). Prints "rows inputs classes".
//   pytorch_reference gradient CONFIG DATA ROWS OUT_DIR [WARMUP]
//       builds the network of CONFIG (initial weights of `init seed=…`), optionally moves it away
//       from its initial weights with WARMUP optimizer steps on the following batches of ROWS
//       positions (rows ROWS to (WARMUP + 1) × ROWS, so the biases are no longer zero), then runs
//       one forward and backward pass on the first ROWS positions of DATA exactly as Network::fit
//       does (zero gradients, forward in training mode, loss gradient, backward without the fused
//       softmax), and writes to OUT_DIR: manifest.txt, weights.bin (the weights of that pass),
//       gradients.bin, batch.x.u8, batch.y.u8, probabilities.bin, loss.txt.
//   pytorch_reference train CONFIG TRAIN VALID EPOCHS OUT_DIR [INDEX]
//       trains the network of CONFIG on TRAIN for EPOCHS epochs with the optimizer, batch size and
//       seed of the configuration (INDEX, 0 by default, is added to both seeds: the initial weights
//       of Network::fromConfig(CONFIG, INDEX) and the shuffling seed; no early stopping, no
//       augmentation, the validation set is only scored, never used to choose anything), and writes
//       to OUT_DIR: manifest.txt, init_weights.bin, final_weights.bin, orders.bin (the row order of
//       every epoch, uint32, so that PyTorch can replay the same batches), epochs.csv,
//       valid_predictions.u8.
//
// Binary files hold little-endian float64 (weights, gradients, probabilities) and uint32 (orders),
// the layout of an x86-64 or ARM machine. A parameter is written row-major, as Matrix stores it, in
// the order of Network::parameters(); manifest.txt has one line per parameter: "layer type index
// rows cols" (layer counted from 0, index 0 = weights and 1 = bias).
//
// Errors go to stderr, exit code 84.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Cli.hpp"
#include "Encoding.hpp"
#include "Fen.hpp"
#include "my_torch/my_torch.hpp"

namespace {

constexpr int EXIT_ERROR = 84;

using my_torch::Error;
using my_torch::Matrix;
using my_torch::Network;

std::size_t count(const std::string &text)
{
    std::size_t used = 0;
    std::size_t value = 0;
    try {
        value = std::stoul(text, &used);
    } catch (const std::exception &) {
        used = 0;
    }
    if (used == 0 || used != text.size())
        throw Error("'" + text + "' is not a whole number");
    return value;
}

// A file of labelled positions, with the classes of the network and the encoding it names.
struct Positions {
    std::vector<analyzer::Sample> samples;
    std::vector<std::size_t> classes;
    const analyzer::Encoding *encoding = nullptr;
    std::size_t classCount = 0;

    Positions(const Network &network, const std::string &path)
    {
        const std::vector<std::string> &labels = network.info().labels;
        samples = analyzer::parseFile(path, analyzer::LabelPolicy::Required);
        for (const analyzer::Sample &s : samples)
            classes.push_back(
                analyzer::labelIndex(std::string(analyzer::labelName(*s.label)), labels));
        encoding = &analyzer::findEncoding(network.info().encoding);
        classCount = labels.size();
    }

    analyzer::PositionSource source() const
    {
        std::vector<std::size_t> everyRow(samples.size());
        std::iota(everyRow.begin(), everyRow.end(), 0);
        return analyzer::PositionSource(samples, classes, classCount, *encoding, everyRow);
    }
};

// A source that remembers the rows fit asked for, in order: the batches of every epoch, one after
// the other. It encodes nothing itself.
class RecordingSource : public my_torch::DataSource {
  public:
    explicit RecordingSource(const my_torch::DataSource &inner) : _inner(inner) {}

    std::size_t size() const override
    {
        return _inner.size();
    }
    std::size_t inputWidth() const override
    {
        return _inner.inputWidth();
    }
    std::size_t targetWidth() const override
    {
        return _inner.targetWidth();
    }
    void fill(const std::vector<std::size_t> &rows, Matrix &inputs, Matrix &targets) const override
    {
        for (const std::size_t r : rows)
            _order.push_back(static_cast<std::uint32_t>(r));
        _inner.fill(rows, inputs, targets);
    }
    const std::vector<std::uint32_t> &order() const
    {
        return _order;
    }

  private:
    const my_torch::DataSource &_inner;
    mutable std::vector<std::uint32_t> _order;
};

std::ofstream openBinary(const std::string &path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw Error("cannot write '" + path + "'");
    return out;
}

void writeManifest(Network &network, const std::string &path)
{
    std::ofstream out(path);
    if (!out)
        throw Error("cannot write '" + path + "'");
    for (std::size_t i = 0; i < network.layers().size(); i++) {
        const auto &layer = network.layers()[i];
        const std::vector<my_torch::Parameter> params = layer->parameters();
        for (std::size_t p = 0; p < params.size(); p++)
            out << i << ' ' << layer->type() << ' ' << p << ' ' << params[p].value->rows() << ' '
                << params[p].value->cols() << '\n';
    }
}

void writeParameters(Network &network, const std::string &path, bool gradients)
{
    std::ofstream out = openBinary(path);
    for (const my_torch::Parameter &p : network.parameters()) {
        const Matrix &m = gradients ? *p.grad : *p.value;
        out.write(reinterpret_cast<const char *>(m.data()),
                  static_cast<std::streamsize>(m.size() * sizeof(my_torch::Scalar)));
    }
}

// The rows of a source [0, rows) as bytes: 0/1 inputs, one class per row.
void writeBytes(const my_torch::DataSource &source, std::size_t rows, const std::string &xPath,
                const std::string &yPath)
{
    std::ofstream x = openBinary(xPath);
    std::ofstream y = openBinary(yPath);
    constexpr std::size_t CHUNK = 4096;
    for (std::size_t begin = 0; begin < rows; begin += CHUNK) {
        const std::size_t end = std::min(begin + CHUNK, rows);
        std::vector<std::size_t> indices(end - begin);
        std::iota(indices.begin(), indices.end(), begin);
        Matrix inputs(indices.size(), source.inputWidth());
        Matrix targets(indices.size(), source.targetWidth());
        source.fill(indices, inputs, targets);
        std::vector<char> bytes(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); i++) {
            const my_torch::Scalar v = inputs.data()[i];
            if (v != 0 && v != 1)
                throw Error("an encoded input is neither 0 nor 1 (" + std::to_string(v) +
                            "): it cannot be stored as a byte");
            bytes[i] = static_cast<char>(v);
        }
        x.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        const std::vector<std::size_t> classes = my_torch::classesOf(targets);
        std::vector<char> labels(classes.begin(), classes.end());
        y.write(labels.data(), static_cast<std::streamsize>(labels.size()));
    }
}

void exportData(const std::vector<std::string> &args)
{
    if (args.size() != 3)
        throw Error("usage: pytorch_reference export CONFIG DATA OUT_PREFIX");
    Network network = Network::fromConfig(args[0], 0);
    const Positions positions(network, args[1]);
    const analyzer::PositionSource source = positions.source();
    writeBytes(source, source.size(), args[2] + ".x.u8", args[2] + ".y.u8");
    std::cout << source.size() << ' ' << source.inputWidth() << ' ' << source.targetWidth() << '\n';
}

void gradient(const std::vector<std::string> &args)
{
    if (args.size() != 4 && args.size() != 5)
        throw Error("usage: pytorch_reference gradient CONFIG DATA ROWS OUT_DIR [WARMUP]");
    const std::string &dir = args[3];
    Network network = Network::fromConfig(args[0], 0);
    const Positions positions(network, args[1]);
    const analyzer::PositionSource source = positions.source();
    const std::size_t rows = count(args[2]);
    if (rows == 0 || rows > source.size())
        throw Error("ROWS must be between 1 and " + std::to_string(source.size()));

    const std::size_t warmup = args.size() == 5 ? count(args[4]) : 0;
    if (rows * (warmup + 1) > source.size())
        throw Error("DATA has " + std::to_string(source.size()) + " positions, " +
                    std::to_string(rows * (warmup + 1)) + " are needed");

    const auto loss = my_torch::makeLoss(network.info().training.loss);
    const auto optimizer = my_torch::makeOptimizer(network.info().training.optimizer);
    const bool skipSoftmax = loss->fusesSoftmax();
    const std::size_t lastBackward =
        skipSoftmax ? network.layers().size() - 1 : network.layers().size();
    writeManifest(network, dir + "/manifest.txt");

    // Rows [first, first + rows) of the source, one forward and backward pass as Network::fit does
    // for one batch; returns the loss and leaves the probabilities in `prediction`.
    Matrix prediction;
    const auto pass = [&](std::size_t first) {
        std::vector<std::size_t> indices(rows);
        std::iota(indices.begin(), indices.end(), first);
        Matrix inputs(rows, source.inputWidth());
        Matrix targets(rows, source.targetWidth());
        source.fill(indices, inputs, targets);
        for (const my_torch::Parameter &p : network.parameters())
            p.grad->fill(0);
        prediction = network.forward(inputs, true);
        const my_torch::Scalar value = loss->value(prediction, targets);
        Matrix grad = loss->gradient(prediction, targets);
        for (std::size_t i = lastBackward; i-- > 0;)
            grad = network.layers()[i]->backward(grad);
        return value;
    };
    for (std::size_t step = 1; step <= warmup; step++) {
        pass(step * rows);
        optimizer->step(network.parameters());
    }
    writeParameters(network, dir + "/weights.bin", false);
    const my_torch::Scalar value = pass(0);

    writeParameters(network, dir + "/gradients.bin", true);
    writeBytes(source, rows, dir + "/batch.x.u8", dir + "/batch.y.u8");
    openBinary(dir + "/probabilities.bin")
        .write(reinterpret_cast<const char *>(prediction.data()),
               static_cast<std::streamsize>(prediction.size() * sizeof(my_torch::Scalar)));
    std::ofstream(dir + "/loss.txt") << std::setprecision(17) << value << '\n';
    std::cerr << "gradient: " << rows << " positions, loss " << std::setprecision(17) << value
              << '\n';
}

struct Evaluation {
    my_torch::Scalar loss = 0;
    my_torch::Scalar accuracy = 0;
    std::vector<std::uint8_t> predicted;
};

// Loss (batch mean, weighted by chunk size) and accuracy of the network on a whole source.
Evaluation evaluate(Network &network, const my_torch::DataSource &source,
                    const my_torch::Loss &loss)
{
    Evaluation result;
    std::size_t right = 0;
    my_torch::Scalar sum = 0;
    for (std::size_t begin = 0; begin < source.size(); begin += my_torch::EVALUATION_CHUNK) {
        const std::size_t end = std::min(begin + my_torch::EVALUATION_CHUNK, source.size());
        std::vector<std::size_t> rows(end - begin);
        std::iota(rows.begin(), rows.end(), begin);
        Matrix inputs(rows.size(), source.inputWidth());
        Matrix targets(rows.size(), source.targetWidth());
        source.fill(rows, inputs, targets);
        const Matrix probabilities = network.predict(inputs);
        sum += loss.value(probabilities, targets) * static_cast<my_torch::Scalar>(rows.size());
        const std::vector<std::size_t> expected = my_torch::classesOf(targets);
        const std::vector<std::size_t> predicted = my_torch::classesOf(probabilities);
        for (std::size_t i = 0; i < rows.size(); i++) {
            right += expected[i] == predicted[i];
            result.predicted.push_back(static_cast<std::uint8_t>(predicted[i]));
        }
    }
    result.loss = sum / static_cast<my_torch::Scalar>(source.size());
    result.accuracy =
        static_cast<my_torch::Scalar>(right) / static_cast<my_torch::Scalar>(source.size());
    return result;
}

void train(const std::vector<std::string> &args)
{
    if (args.size() != 5 && args.size() != 6)
        throw Error("usage: pytorch_reference train CONFIG TRAIN VALID EPOCHS OUT_DIR [INDEX]");
    const std::string &dir = args[4];
    const std::size_t epochs = count(args[3]);
    if (epochs == 0)
        throw Error("EPOCHS must be at least 1");
    const std::size_t index = args.size() == 6 ? count(args[5]) : 0;
    Network network = Network::fromConfig(args[0], index);
    const Positions trainPositions(network, args[1]);
    const Positions validPositions(network, args[2]);
    const analyzer::PositionSource trainSource = trainPositions.source();
    const analyzer::PositionSource validSource = validPositions.source();
    const RecordingSource recording(trainSource);

    const my_torch::TrainingSpec &spec = network.info().training;
    my_torch::TrainConfig config = spec.config;
    config.epochs = epochs;
    config.seed += index;
    config.patience = 0;
    config.validation = 0;
    config.validationTail = 0;
    const auto loss = my_torch::makeLoss(spec.loss);
    const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
    const auto scheduler = my_torch::makeScheduler(spec.scheduler);

    writeManifest(network, dir + "/manifest.txt");
    writeParameters(network, dir + "/init_weights.bin", false);

    // The seconds of an epoch are those of fit between two callbacks; the scoring of the validation
    // set inside the callback is left out (the mark is taken again after it).
    std::ofstream csv(dir + "/epochs.csv");
    csv << "epoch,seconds,train_loss,train_accuracy,val_loss,val_accuracy\n";
    auto mark = std::chrono::steady_clock::now();
    config.onEpoch = [&](std::size_t epoch, const my_torch::EpochStats &stats) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - mark).count();
        const Evaluation valid = evaluate(network, validSource, *loss);
        csv << std::setprecision(17) << epoch + 1 << ',' << seconds << ',' << stats.trainLoss << ','
            << stats.trainAccuracy << ',' << valid.loss << ',' << valid.accuracy << std::endl;
        std::cerr << "epoch " << epoch + 1 << ": " << seconds << " s, train loss "
                  << stats.trainLoss << ", validation accuracy " << valid.accuracy << '\n';
        mark = std::chrono::steady_clock::now();
    };
    network.fit(recording, nullptr, *loss, *optimizer, config, scheduler.get());

    writeParameters(network, dir + "/final_weights.bin", false);
    const std::vector<std::uint32_t> &order = recording.order();
    openBinary(dir + "/orders.bin")
        .write(reinterpret_cast<const char *>(order.data()),
               static_cast<std::streamsize>(order.size() * sizeof(std::uint32_t)));
    const Evaluation valid = evaluate(network, validSource, *loss);
    openBinary(dir + "/valid_predictions.u8")
        .write(reinterpret_cast<const char *>(valid.predicted.data()),
               static_cast<std::streamsize>(valid.predicted.size()));
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const std::vector<std::string> args(argv + 1, argv + argc);
        if (args.empty())
            throw Error("usage: pytorch_reference export|gradient|train ...");
        const std::vector<std::string> rest(args.begin() + 1, args.end());
        if (args[0] == "export")
            exportData(rest);
        else if (args[0] == "gradient")
            gradient(rest);
        else if (args[0] == "train")
            train(rest);
        else
            throw Error("unknown command '" + args[0] + "'");
    } catch (const std::exception &e) {
        std::cerr << "pytorch_reference: " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
