// my_torch_ensemble: builds an ensemble file (#40, ADR-0045) that my_torch_analyzer --predict reads
// as its LOADFILE.
//   average|vote OUT NETWORK[:WEIGHT]...   combines the networks (weight 1 by default)
//   stack OUT CONFIG CHESSFILE NETWORK...  trains the combiner described by CONFIG (a network
//                                          configuration, ADR-0013, whose input is the members'
//                                          probabilities side by side) on the labelled positions
//                                          of CHESSFILE, which the members must not have learned
// Errors on stderr, exit code 84.
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "Encoding.hpp"
#include "Fen.hpp"
#include "my_torch/my_torch.hpp"

using my_torch::Ensemble;
using my_torch::Error;
using my_torch::Matrix;
using my_torch::Network;

namespace {

constexpr int EXIT_ERROR = 84;
constexpr std::size_t CHUNK = 4096;

const char *usage =
    "USAGE\n"
    "    ./my_torch_ensemble average|vote OUT NETWORK[:WEIGHT]...\n"
    "    ./my_torch_ensemble stack OUT CONFIG CHESSFILE NETWORK...\n\n"
    "DESCRIPTION\n"
    "    Writes into OUT an ensemble of the networks, which my_torch_analyzer --predict reads\n"
    "    like a network file. average: weighted mean of the probabilities; vote: weighted\n"
    "    majority of the predicted classes; stack: a combiner network, described by CONFIG and\n"
    "    trained on the labelled positions of CHESSFILE, reads the members' probabilities.\n";

// "file.nn:2.5" → ("file.nn", 2.5); no ":" → weight 1.
std::pair<std::string, my_torch::Scalar> memberArgument(const std::string &arg)
{
    const std::size_t colon = arg.rfind(':');
    if (colon == std::string::npos)
        return {arg, 1};
    try {
        std::size_t used = 0;
        const double weight = std::stod(arg.substr(colon + 1), &used);
        if (used == arg.size() - colon - 1)
            return {arg.substr(0, colon), weight};
    } catch (const std::exception &) {
    }
    throw Error("'" + arg + "': expected NETWORK or NETWORK:WEIGHT");
}

const analyzer::Encoding &encodingOf(const Network &net)
{
    const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
    if (encoding.width() != net.info().inputs)
        throw Error("encoding '" + encoding.name() + "' does not match the network's inputs");
    return encoding;
}

void trainCombiner(Ensemble &ensemble, const std::string &config, const std::string &chessfile)
{
    Network combiner = Network::fromConfig(config);
    const std::vector<analyzer::Sample> samples =
        analyzer::parseFile(chessfile, analyzer::LabelPolicy::Required);
    const auto &labels = ensemble.labels();
    std::vector<const analyzer::Encoding *> encodings;
    for (std::size_t i = 0; i < ensemble.size(); i++)
        encodings.push_back(&encodingOf(ensemble.member(i)));

    // The members' probabilities of every position, and the expected class as one-hot targets.
    my_torch::Dataset all{Matrix(samples.size(), ensemble.size() * labels.size()),
                          Matrix(samples.size(), labels.size())};
    for (std::size_t begin = 0; begin < samples.size(); begin += CHUNK) {
        const std::size_t end = std::min(begin + CHUNK, samples.size());
        std::vector<Matrix> rows;
        for (const analyzer::Encoding *encoding : encodings)
            rows.push_back(analyzer::encodeRows(*encoding, samples, begin, end));
        const Matrix joined = ensemble.memberOutputs(rows);
        std::copy_n(joined.data(), joined.rows() * joined.cols(),
                    all.inputs.data() + begin * joined.cols());
    }
    for (std::size_t r = 0; r < samples.size(); r++) {
        const std::string name(analyzer::labelName(*samples[r].label));
        const auto found = std::find(labels.begin(), labels.end(), name);
        if (found == labels.end())
            throw Error(chessfile + ": position " + std::to_string(r + 1) + ": label '" + name +
                        "' is not one of the members' labels");
        all.targets(r, static_cast<std::size_t>(found - labels.begin())) = 1;
    }

    // Early stopping on a held-out fraction of CHESSFILE when the configuration asks for one.
    const my_torch::TrainingSpec &spec = combiner.info().training;
    const auto order = my_torch::permutation(samples.size(), spec.config.seed);
    const auto held =
        static_cast<std::size_t>(spec.config.validation * static_cast<double>(samples.size()));
    const auto subset = [&](std::size_t from, std::size_t to) {
        my_torch::Dataset part{Matrix(to - from, all.inputs.cols()),
                               Matrix(to - from, all.targets.cols())};
        for (std::size_t r = from; r < to; r++) {
            std::copy_n(&all.inputs(order[r], 0), all.inputs.cols(), &part.inputs(r - from, 0));
            std::copy_n(&all.targets(order[r], 0), all.targets.cols(), &part.targets(r - from, 0));
        }
        return part;
    };
    const my_torch::Dataset validation = subset(0, held);
    const my_torch::Dataset train = subset(held, samples.size());
    const auto loss = my_torch::makeLoss(spec.loss);
    const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
    const auto scheduler = my_torch::makeScheduler(spec.scheduler);
    const my_torch::History history = combiner.fit(train, held ? &validation : nullptr, *loss,
                                                   *optimizer, spec.config, scheduler.get());
    const my_torch::EpochStats &last = history.back();
    std::cout << "combiner: " << history.size() << " epochs on " << train.inputs.rows()
              << " positions, loss " << last.trainLoss << ", accuracy " << last.trainAccuracy;
    if (held)
        std::cout << ", held-out loss " << last.validationLoss << ", accuracy "
                  << last.validationAccuracy;
    std::cout << '\n';
    ensemble.setCombiner(std::move(combiner));
}

int run(const std::vector<std::string> &args)
{
    if (args.size() < 3)
        throw Error("missing arguments");
    const my_torch::Combination combination = my_torch::combinationFromName(args[0]);
    const std::string &out = args[1];
    const bool stack = combination == my_torch::Combination::Stack;
    const std::size_t first = stack ? 4 : 2;
    if (args.size() <= first)
        throw Error("expected at least one NETWORK");

    Ensemble ensemble(combination);
    for (std::size_t i = first; i < args.size(); i++) {
        const auto [path, weight] = memberArgument(args[i]);
        if (stack && weight != 1)
            throw Error("stack learns how to combine the members: give no weights");
        ensemble.add(Network::load(path), weight);
    }
    if (stack)
        trainCombiner(ensemble, args[2], args[3]);
    ensemble.save(out);
    std::cout << out << ": " << my_torch::combinationName(combination) << " of " << ensemble.size()
              << " networks\n";
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() == 1 && (args[0] == "-h" || args[0] == "--help")) {
        std::cout << usage;
        return 0;
    }
    try {
        return run(args);
    } catch (const std::exception &error) {
        std::cerr << "my_torch_ensemble: " << error.what() << '\n' << usage;
        return EXIT_ERROR;
    }
}
