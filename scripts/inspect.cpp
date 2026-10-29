// Network inspection (#36, docs/performance/19-inspection.md, ADR-0035).
// Usage:
//   inspect summary NETWORK
//       prints modelSummary.
//   inspect histograms NETWORK TRAINING_FILE OUTPUT_CSV
//       trains NETWORK (a generated, untrained network) on TRAINING_FILE as my_torch_analyzer
//       --train does (validation_tail lines at the end), and writes after epoch 0 (the initial
//       weights) and every epoch the histogram of the weights and of the gradients of the last
//       batch of each Dense layer: epoch,layer,kind,bin,low,high,count.
//   inspect saliency NETWORK LABELLED_FILE OUTPUT_CSV
//       for every position in check or mate, the saliency of its squares (gui::saliency) and the
//       squares the rule-based oracle names: index,expected,predicted,probability,king,checkers,
//       top_occlusion,top_gradient, then the 64 occlusion values (a8 to h1).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Cli.hpp"
#include "Encoding.hpp"
#include "Fen.hpp"
#include "Oracle.hpp"
#include "Saliency.hpp"
#include "my_torch/my_torch.hpp"

namespace {

using my_torch::Matrix;
using my_torch::Scalar;

constexpr int BINS = 40;

// Weights: 40 bins over [-0.5, 0.5]; gradients: 40 bins of log10 |g| over [-10, 0]. Values out of
// range go to the first or last bin.
void histogram(std::ostream &out, std::size_t epoch, std::size_t layer, const char *kind,
               const Matrix &m, bool logarithmic)
{
    const Scalar low = logarithmic ? -10 : -0.5;
    const Scalar high = logarithmic ? 0 : 0.5;
    std::vector<long> counts(BINS, 0);
    for (std::size_t i = 0; i < m.size(); i++) {
        Scalar v = m.data()[i];
        if (logarithmic)
            v = std::log10(std::max(std::abs(v), 1e-300));
        const int bin = static_cast<int>(std::floor((v - low) / (high - low) * BINS));
        counts[static_cast<std::size_t>(std::clamp(bin, 0, BINS - 1))]++;
    }
    for (int b = 0; b < BINS; b++)
        out << epoch << ',' << layer << ',' << kind << ',' << b << ','
            << low + (high - low) * b / BINS << ',' << low + (high - low) * (b + 1) / BINS << ','
            << counts[static_cast<std::size_t>(b)] << '\n';
}

void writeHistograms(std::ostream &out, my_torch::Network &net, std::size_t epoch)
{
    for (std::size_t i = 0; i < net.layers().size(); i++) {
        auto *dense = dynamic_cast<my_torch::Dense *>(net.layers()[i].get());
        if (!dense)
            continue;
        histogram(out, epoch, i + 1, "weight", dense->weights(), false);
        if (epoch > 0)
            histogram(out, epoch, i + 1, "gradient", *dense->parameters()[0].grad, true);
    }
}

int histograms(const std::string &networkPath, const std::string &trainingFile,
               const std::string &output)
{
    my_torch::Network net = my_torch::Network::load(networkPath);
    const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
    const my_torch::TrainingSpec spec = net.info().training;
    const auto &labels = net.info().labels;
    const std::vector<analyzer::Sample> samples =
        analyzer::parseFile(trainingFile, analyzer::LabelPolicy::Required);
    std::vector<std::size_t> classes;
    for (const analyzer::Sample &s : samples)
        classes.push_back(
            analyzer::labelIndex(std::string(analyzer::labelName(*s.label)), labels));
    const std::size_t tail = spec.config.validationTail;
    if (tail == 0 || tail >= samples.size())
        throw my_torch::Error("histograms: the network needs validation_tail (ADR-0022)");
    std::vector<std::size_t> rows(samples.size());
    std::iota(rows.begin(), rows.end(), 0);
    const auto split = rows.end() - static_cast<std::ptrdiff_t>(tail);
    const analyzer::PositionSource train(samples, classes, labels.size(), encoding,
                                         std::vector<std::size_t>(rows.begin(), split));
    const analyzer::PositionSource validation(samples, classes, labels.size(), encoding,
                                              std::vector<std::size_t>(split, rows.end()));

    std::ofstream out(output);
    out << "epoch,layer,kind,bin,low,high,count\n";
    writeHistograms(out, net, 0);
    my_torch::TrainConfig config = spec.config;
    config.onEpoch = [&](std::size_t epoch, const my_torch::EpochStats &stats) {
        writeHistograms(out, net, epoch + 1);
        std::cout << "epoch " << epoch + 1 << "  loss " << stats.trainLoss << "  validation loss "
                  << stats.validationLoss << "  validation accuracy " << stats.validationAccuracy
                  << std::endl;
    };
    const auto loss = my_torch::makeLoss(spec.loss);
    const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
    const auto scheduler = my_torch::makeScheduler(spec.scheduler);
    net.fit(train, &validation, *loss, *optimizer, config, scheduler.get());
    return 0;
}

int saliency(const std::string &networkPath, const std::string &labelled,
             const std::string &output)
{
    my_torch::Network net = my_torch::Network::load(networkPath);
    const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
    const auto &labels = net.info().labels;
    const std::vector<analyzer::Sample> samples =
        analyzer::parseFile(labelled, analyzer::LabelPolicy::Required);
    std::ofstream out(output);
    out.precision(5);
    out << "index,expected,predicted,probability,king,checkers,top_occlusion,top_gradient";
    for (int s = 0; s < 64; s++)
        out << ",o" << s;
    out << '\n';
    for (std::size_t i = 0; i < samples.size(); i++) {
        const analyzer::Position &p = samples[i].position;
        int king = -1;
        std::vector<int> checkers;
        for (const analyzer::Color color : {analyzer::Color::White, analyzer::Color::Black})
            if (oracle::inCheck(p, color)) {
                king = oracle::findKing(p, color);
                checkers = oracle::attackers(p, king, oracle::opponent(color));
            }
        if (king < 0)
            continue;
        const gui::SquareSaliency s = gui::saliency(net, encoding, p);
        const auto top = [](const std::array<Scalar, 64> &v) {
            return std::max_element(v.begin(), v.end()) - v.begin();
        };
        std::string names;
        for (const int c : checkers)
            names += (names.empty() ? "" : " ") + std::to_string(c);
        out << i << ','
            << analyzer::labelIndex(std::string(analyzer::labelName(*samples[i].label)), labels)
            << ',' << s.predicted << ',' << s.probability << ',' << king << ',' << names << ','
            << top(s.occlusion) << ',' << top(s.gradient);
        for (const Scalar v : s.occlusion)
            out << ',' << v;
        out << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.size() == 2 && args[0] == "summary") {
            my_torch::Network net = my_torch::Network::load(args[1]);
            std::cout << my_torch::modelSummary(net);
            return 0;
        }
        if (args.size() == 4 && args[0] == "histograms")
            return histograms(args[1], args[2], args[3]);
        if (args.size() == 4 && args[0] == "saliency")
            return saliency(args[1], args[2], args[3]);
        std::cerr << "usage: inspect summary NETWORK | histograms NETWORK TRAINING_FILE OUT | "
                     "saliency NETWORK LABELLED_FILE OUT\n";
    } catch (const std::exception &e) {
        std::cerr << "inspect: " << e.what() << '\n';
    }
    return 84;
}
