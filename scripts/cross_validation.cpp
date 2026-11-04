// Stratified k-fold cross-validation of a chess network (#39, docs/performance/21-cross-validation.md,
// ADR-0039): the objective for my_torch's crossValidate. Each fold trains a fresh network built from
// CONFIG on the training part for a fixed number of epochs, and scores it on the held-out part with
// my_torch::report. There is no early stopping on the held-out part (it would pick the model with
// the labels it is scored on) and no augmentation (the symmetric images of a held-out position
// would be in the training part). DATA is read whole: pass it a part of the training split, never
// the test split.
//
//   cross_validation CONFIG DATA K SEED EPOCHS OUT_CSV MODEL_NAME
//
// SEED draws the folds. Appends one line per fold to OUT_CSV (the header is written when the file
// does not exist): model, fold, train_rows, validation_rows, epochs, seconds, accuracy,
// macro_precision, macro_recall, macro_f1. Prints the mean ± standard deviation of each metric to
// stderr. Errors go to stderr, exit code 84.
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
using my_torch::Scalar;

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

// The network's report on a source, by my_torch::report (ADR-0026).
my_torch::ClassificationReport score(my_torch::Network &net, const my_torch::DataSource &source)
{
    std::vector<std::size_t> expected;
    std::vector<std::size_t> predicted;
    for (std::size_t begin = 0; begin < source.size(); begin += my_torch::EVALUATION_CHUNK) {
        const std::size_t end = std::min(begin + my_torch::EVALUATION_CHUNK, source.size());
        std::vector<std::size_t> rows(end - begin);
        std::iota(rows.begin(), rows.end(), begin);
        my_torch::Matrix inputs(rows.size(), source.inputWidth());
        my_torch::Matrix targets(rows.size(), source.targetWidth());
        source.fill(rows, inputs, targets);
        for (const std::size_t c : my_torch::classesOf(targets))
            expected.push_back(c);
        for (const std::size_t c : my_torch::classesOf(net.predict(inputs)))
            predicted.push_back(c);
    }
    return my_torch::report(my_torch::confusion(expected, predicted, source.targetWidth()));
}

void print(const char *name, const my_torch::MetricSummary &s)
{
    std::cerr << "  " << name << ": " << s.mean << " ± " << s.std << '\n';
}

void run(const std::vector<std::string> &args)
{
    if (args.size() != 7)
        throw Error("usage: cross_validation CONFIG DATA K SEED EPOCHS OUT_CSV MODEL_NAME");
    const std::string &config = args[0];
    const std::size_t k = count(args[2]);
    const std::size_t seed = count(args[3]);
    const std::size_t epochs = count(args[4]);
    const std::string &csv = args[5];
    const std::string &model = args[6];
    if (epochs == 0)
        throw Error("EPOCHS must be at least 1");

    const std::vector<std::string> labels =
        my_torch::Network::fromConfig(config, 0).info().labels;
    const std::vector<analyzer::Sample> samples =
        analyzer::parseFile(args[1], analyzer::LabelPolicy::Required);
    std::vector<std::size_t> classes;
    for (const analyzer::Sample &s : samples)
        classes.push_back(analyzer::labelIndex(std::string(analyzer::labelName(*s.label)), labels));
    std::vector<std::size_t> perClass(labels.size(), 0);
    for (const std::size_t c : classes)
        perClass[c]++;
    for (std::size_t c = 0; c < labels.size(); c++)
        if (perClass[c] < k)
            throw Error("class '" + labels[c] + "' has " + std::to_string(perClass[c]) +
                        " rows in " + args[1] + ", fewer than the " + std::to_string(k) +
                        " folds: use more data");

    const analyzer::Encoding &encoding =
        analyzer::findEncoding(my_torch::Network::fromConfig(config, 0).info().encoding);
    std::vector<std::size_t> everyRow(samples.size());
    std::iota(everyRow.begin(), everyRow.end(), 0);
    const analyzer::PositionSource source(samples, classes, labels.size(), encoding, everyRow);

    if (!std::ifstream(csv))
        std::ofstream(csv) << "model,fold,train_rows,validation_rows,epochs,seconds,accuracy,"
                              "macro_precision,macro_recall,macro_f1\n";

    const auto trainer = [&](const my_torch::DataSource &train, const my_torch::DataSource &held,
                             std::size_t fold) {
        const auto start = std::chrono::steady_clock::now();
        my_torch::Network net = my_torch::Network::fromConfig(config, 0);
        const my_torch::TrainingSpec &spec = net.info().training;
        my_torch::TrainConfig trainConfig = spec.config;
        trainConfig.epochs = epochs;
        trainConfig.patience = 0;
        trainConfig.validation = 0;
        trainConfig.validationTail = 0;
        const auto loss = my_torch::makeLoss(spec.loss);
        const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
        const auto scheduler = my_torch::makeScheduler(spec.scheduler);
        net.fit(train, nullptr, *loss, *optimizer, trainConfig, scheduler.get());
        const my_torch::ClassificationReport report = score(net, held);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream(csv, std::ios::app)
            << std::setprecision(10) << model << ',' << fold << ',' << train.size() << ','
            << held.size() << ',' << epochs << ',' << seconds << ',' << report.accuracy << ','
            << report.macroPrecision << ',' << report.macroRecall << ',' << report.macroF1 << '\n';
        std::cerr << model << " fold " << fold << ": accuracy " << report.accuracy
                  << ", macro F1 " << report.macroF1 << ", " << seconds << " s\n";
        return report;
    };
    const my_torch::CrossValidation result = my_torch::crossValidate(source, k, seed, trainer);

    std::cerr << model << ", " << k << " folds (mean ± standard deviation):\n";
    print("accuracy", result.accuracy);
    print("macro precision", result.macroPrecision);
    print("macro recall", result.macroRecall);
    print("macro F1", result.macroF1);
}

} // namespace

int main(int argc, char **argv)
{
    try {
        run(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception &e) {
        std::cerr << "cross_validation: " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
