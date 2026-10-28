// Hyperparameter search of the chess network (#10, docs/performance/15-hyperparameters.md,
// ADR-0028): the objective for my_torch's search module. Each trial writes a configuration from
// TEMPLATE with its values, trains it on TRAIN with early stopping on VALIDATION (the same path as
// my_torch_analyzer --train), and returns the macro F1 on VALIDATION computed by my_torch::report.
// The test split is never read.
//
//   hyperparam_search TEMPLATE TRAIN VALIDATION OUTPUT_DIRECTORY MODE
//     grid                every combination of the space; log OUTPUT_DIRECTORY/grid.csv
//     random COUNT SEED   COUNT combinations drawn with SEED; reuses the trials of grid.csv
//                         (random_SEED.csv in the order of the draw; log random_SEED_log.csv)
//     seeds TOP SEEDS     the TOP best combinations of grid.csv again with SEEDS training seeds
//                         each (2 to SEEDS + 1; seed 1 is the grid run); log seeds.csv
//     trial INDEX         one combination of the grid, no log (to time one trial)
//
// Space: lr {0.0003, 0.001, 0.003} × batch {32, 64, 128} × hidden {256x128, 512x256} ×
// dropout {0, 0.2}, 36 combinations. Every trial also appends its epochs, seconds and validation
// accuracy to OUTPUT_DIRECTORY/details.csv. Logs make every mode resumable after a stop.
// Errors go to stderr, exit code 84.
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
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

struct LabelledFile {
    std::vector<analyzer::Sample> samples;
    std::vector<std::size_t> classes;
};

std::string readText(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw Error("cannot open '" + path + "'");
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

my_torch::SearchSpace space()
{
    my_torch::SearchSpace s;
    s.add("lr", {0.0003, 0.001, 0.003});
    s.add("batch", {32.0, 64.0, 128.0});
    s.add("hidden", {std::string("256x128"), std::string("512x256")});
    s.add("dropout", {0.0, 0.2});
    return s;
}

// The template's configuration with the trial's values: optimizer rate, batch size, training and
// initialisation seed, and the hidden layers (each followed by ReLU, then dropout when not 0).
std::string configFor(const std::string &base, const my_torch::Trial &t, std::size_t seed)
{
    std::string text = std::regex_replace(base, std::regex(R"(lr=[0-9.e-]+)"),
                                          "lr=" + my_torch::toString(t.number("lr")));
    text = std::regex_replace(text, std::regex(R"(batch=[0-9]+)"),
                              "batch=" + std::to_string(t.integer("batch")));
    text = std::regex_replace(text, std::regex(R"((\ntrain [^\n]*) seed=[0-9]+)"),
                              "$1 seed=" + std::to_string(42 + seed - 1));
    text = std::regex_replace(text, std::regex(R"(init seed=[0-9]+)"),
                              "init seed=" + std::to_string(seed));
    const std::size_t layers = text.find("\nlayers ");
    const std::size_t inputStart = text.find("\ninput ");
    if (layers == std::string::npos || inputStart == std::string::npos)
        throw Error("template: needs an 'input' line and a 'layers' section");
    std::istringstream input(text.substr(inputStart + 1));
    std::string word;
    std::string encoding;
    std::size_t width = 0;
    input >> word >> encoding >> width;
    std::vector<std::size_t> sizes = {width};
    std::istringstream hidden(t.text("hidden"));
    for (std::string part; std::getline(hidden, part, 'x');)
        sizes.push_back(std::stoul(part));
    sizes.push_back(5);
    const Scalar dropout = t.number("dropout");
    std::vector<std::string> lines;
    for (std::size_t i = 0; i + 1 < sizes.size(); i++) {
        lines.push_back("dense " + std::to_string(sizes[i]) + " " + std::to_string(sizes[i + 1]));
        if (i + 2 < sizes.size()) {
            lines.push_back("relu");
            if (dropout > 0)
                lines.push_back("dropout rate=" + my_torch::toString(dropout));
        }
    }
    lines.push_back("softmax");
    std::string block = "\nlayers " + std::to_string(lines.size()) + "\n";
    for (const std::string &line : lines)
        block += line + "\n";
    return text.substr(0, layers) + block;
}

class Chess {
  public:
    Chess(std::string base, const std::string &train, const std::string &validation,
          std::string out)
        : _base(std::move(base)), _out(std::move(out))
    {
        _labels = network(my_torch::combination(space(), 0), 1).info().labels;
        _train = read(train);
        _validation = read(validation);
        if (!std::ifstream(_out + "/details.csv"))
            std::ofstream(_out + "/details.csv") << "index,seed,epochs,seconds,accuracy,macro_f1\n";
    }

    Scalar operator()(const my_torch::Trial &t, std::size_t seed)
    {
        const auto start = std::chrono::steady_clock::now();
        my_torch::Network net = network(t, seed);
        const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
        const my_torch::TrainingSpec &spec = net.info().training;
        const analyzer::PositionSource train(_train.samples, _train.classes, _labels.size(),
                                             encoding, all(_train.samples.size()));
        const analyzer::PositionSource validation(_validation.samples, _validation.classes,
                                                  _labels.size(), encoding,
                                                  all(_validation.samples.size()));
        const auto loss = my_torch::makeLoss(spec.loss);
        const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
        const auto scheduler = my_torch::makeScheduler(spec.scheduler);
        const my_torch::History history =
            net.fit(train, &validation, *loss, *optimizer, spec.config, scheduler.get());
        const my_torch::ClassificationReport report = score(net, validation);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream details(_out + "/details.csv", std::ios::app);
        details << t.index() << ',' << seed << ',' << history.size() << ',' << seconds << ','
                << report.accuracy << ',' << report.macroF1 << '\n';
        std::cerr << "trial " << t.index() << " seed " << seed << ": macro F1 " << report.macroF1
                  << ", " << history.size() << " epochs, " << seconds << " s\n";
        return report.macroF1;
    }

  private:
    static std::vector<std::size_t> all(std::size_t n)
    {
        std::vector<std::size_t> rows(n);
        std::iota(rows.begin(), rows.end(), 0);
        return rows;
    }

    my_torch::Network network(const my_torch::Trial &t, std::size_t seed) const
    {
        const std::string path = _out + "/trial.conf";
        std::ofstream(path) << configFor(_base, t, seed);
        return my_torch::Network::fromConfig(path, 0);
    }

    LabelledFile read(const std::string &path) const
    {
        LabelledFile f{analyzer::parseFile(path, analyzer::LabelPolicy::Required), {}};
        for (const analyzer::Sample &s : f.samples)
            f.classes.push_back(
                analyzer::labelIndex(std::string(analyzer::labelName(*s.label)), _labels));
        return f;
    }

    // The network's report on a source, by my_torch::report (ADR-0026).
    static my_torch::ClassificationReport score(my_torch::Network &net,
                                                const my_torch::DataSource &source)
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

    std::string _base;
    std::string _out;
    std::vector<std::string> _labels;
    LabelledFile _train;
    LabelledFile _validation;
};

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

void run(const std::vector<std::string> &args)
{
    if (args.size() < 5)
        throw Error("usage: hyperparam_search TEMPLATE TRAIN VALIDATION OUTPUT_DIRECTORY "
                    "grid | random COUNT SEED | seeds TOP SEEDS | trial INDEX");
    const std::string &out = args[3];
    const std::string &mode = args[4];
    const bool known = (mode == "grid" && args.size() == 5) ||
                       (mode == "trial" && args.size() == 6) ||
                       ((mode == "random" || mode == "seeds") && args.size() == 7);
    if (!known)
        throw Error("unknown mode or wrong number of arguments for '" + mode + "'");
    Chess chess(readText(args[0]), args[1], args[2], out);
    const my_torch::SearchSpace s = space();
    const std::string grid = out + "/grid.csv";
    const auto atSeed1 = [&](const my_torch::Trial &t) { return chess(t, 1); };

    if (mode == "trial") {
        const my_torch::Trial t = my_torch::combination(s, count(args[5]));
        const Scalar macroF1 = chess(t, 1);
        std::cout << "trial " << t.index() << ": macro F1 " << macroF1 << '\n';
    } else if (mode == "grid") {
        const my_torch::SearchResult r = my_torch::GridSearch().run(s, atSeed1, grid);
        std::cout << "best: trial " << r.best().trial.index() << ", macro F1 " << r.best().score
                  << '\n';
    } else if (mode == "random") {
        const std::string log = out + "/random_" + args[6] + "_log.csv";
        // Starts from the grid's trials: a combination already trained is read, not rerun. The
        // log keeps the grid's order; random_SEED.csv lists the trials in the order of the draw.
        if (std::ifstream(grid) && !std::ifstream(log))
            std::ofstream(log) << readText(grid);
        my_torch::RandomSearch(count(args[5]), count(args[6]))
            .run(s, atSeed1, log)
            .saveCsv(out + "/random_" + args[6] + ".csv");
    } else {
        const my_torch::SearchResult r = my_torch::GridSearch().run(s, atSeed1, grid);
        std::vector<my_torch::TrialResult> ranked = r.trials();
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const auto &a, const auto &b) { return a.score > b.score; });
        const std::size_t top = std::min(count(args[5]), ranked.size());
        my_torch::SearchSpace seeds;
        std::vector<my_torch::SearchValue> finalists;
        for (std::size_t i = 0; i < top; i++)
            finalists.push_back(static_cast<Scalar>(ranked[i].trial.index()));
        seeds.add("trial", finalists);
        std::vector<my_torch::SearchValue> seedValues;
        for (std::size_t k = 2; k <= count(args[6]) + 1; k++)
            seedValues.push_back(static_cast<Scalar>(k));
        seeds.add("seed", seedValues);
        my_torch::GridSearch().run(
            seeds,
            [&](const my_torch::Trial &t) {
                return chess(my_torch::combination(s, t.integer("trial")), t.integer("seed"));
            },
            out + "/seeds.csv");
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        run(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception &e) {
        std::cerr << "hyperparam_search: " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
