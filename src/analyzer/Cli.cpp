#include "Cli.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>

#include "Encoding.hpp"
#include "Fen.hpp"
#include "my_torch/my_torch.hpp"

namespace analyzer {

namespace {

using my_torch::Error;

constexpr const char *usage =
    "USAGE\n"
    "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n"
    "\n"
    "DESCRIPTION\n"
    "    --train     Launch the neural network in training mode. Each chessboard in CHESSFILE\n"
    "                must contain inputs to send to the neural network in FEN notation and the\n"
    "                expected output separated by space. If specified, the newly trained neural\n"
    "                network will be saved in SAVEFILE. Otherwise, it will be saved in the\n"
    "                original LOADFILE.\n"
    "    --predict   Launch the neural network in prediction mode. Each chessboard in CHESSFILE\n"
    "                must contain inputs to send to the neural network in FEN notation, and\n"
    "                optionally an expected output.\n"
    "    --save      Save neural network into SAVEFILE. Only works in train mode.\n"
    "    LOADFILE    File containing an artificial neural network\n"
    "    CHESSFILE   File containing chessboards\n"
    "\n"
    "EXTRA OPTION\n"
    "    --checkpoint FILE  In train mode, save the training state into FILE after every\n"
    "                epoch; if FILE exists, resume the training from it.\n"
    "    --progress-every N  In train mode, every N epochs print the epoch's measurements on\n"
    "                stderr and save the weights into\n"
    "                SAVEFILE.YYYYMMDD-HHMMSS.epochN.nn (the time of the save).\n";

struct Options {
    bool train = false;
    std::optional<std::string> save;
    std::optional<std::string> checkpoint;
    std::size_t progressEvery = 0;
    std::string network;
    std::string chessboards;
};

Options parseArguments(const std::vector<std::string> &args)
{
    Options o;
    bool predict = false;
    std::vector<std::string> files;

    for (std::size_t i = 0; i < args.size(); i++) {
        const std::string &a = args[i];
        if (a == "--predict" || a == "--train") {
            if (predict || o.train)
                throw Error("give exactly one of --predict and --train");
            (a == "--train" ? o.train : predict) = true;
        } else if (a == "--save") {
            if (o.save || i + 1 >= args.size())
                throw Error("--save needs exactly one SAVEFILE");
            o.save = args[++i];
        } else if (a == "--checkpoint") {
            if (o.checkpoint || i + 1 >= args.size())
                throw Error("--checkpoint needs exactly one FILE");
            o.checkpoint = args[++i];
        } else if (a == "--progress-every") {
            if (o.progressEvery != 0 || i + 1 >= args.size())
                throw Error("--progress-every needs exactly one positive number");
            const std::string &value = args[++i];
            if (value.find_first_not_of("0123456789") != std::string::npos || value.empty() ||
                value.size() > 9 || std::stoul(value) == 0)
                throw Error("--progress-every needs exactly one positive number");
            o.progressEvery = std::stoul(value);
        } else if (a.size() > 1 && a[0] == '-') {
            throw Error("unknown option '" + a + "'");
        } else {
            files.push_back(a);
        }
    }
    if (!predict && !o.train)
        throw Error("give exactly one of --predict and --train");
    if (o.save && !o.train)
        throw Error("--save only works with --train");
    if (o.checkpoint && !o.train)
        throw Error("--checkpoint only works with --train");
    if (o.progressEvery != 0 && !o.train)
        throw Error("--progress-every only works with --train");
    if (files.size() != 2)
        throw Error("expected LOADFILE and CHESSFILE");
    o.network = files[0];
    o.chessboards = files[1];
    return o;
}

const Encoding &encodingOf(const my_torch::Network &net)
{
    const Encoding &encoding = findEncoding(net.info().encoding);
    if (encoding.width() != net.info().inputs)
        throw Error("encoding '" + encoding.name() + "' gives " + std::to_string(encoding.width()) +
                    " inputs but the network takes " + std::to_string(net.info().inputs));
    return encoding;
}

// LOADFILE is an ensemble (ADR-0045): each member reads the positions in its own encoding.
int predictEnsemble(const Options &o)
{
    my_torch::Ensemble ensemble = my_torch::Ensemble::load(o.network);
    std::vector<const Encoding *> encodings;
    for (std::size_t i = 0; i < ensemble.size(); i++)
        encodings.push_back(&encodingOf(ensemble.member(i)));
    const std::vector<Sample> samples = parseFile(o.chessboards, LabelPolicy::Optional);
    const auto &labels = ensemble.labels();

    for (std::size_t begin = 0; begin < samples.size(); begin += PREDICT_CHUNK) {
        const std::size_t end = std::min(begin + PREDICT_CHUNK, samples.size());
        std::vector<my_torch::Matrix> rows;
        for (const Encoding *encoding : encodings)
            rows.push_back(encodeRows(*encoding, samples, begin, end));
        for (const std::size_t c : ensemble.classify(rows))
            std::cout << labels[c] << '\n';
    }
    return 0;
}

int predict(const Options &o)
{
    if (my_torch::Ensemble::isEnsembleFile(o.network))
        return predictEnsemble(o);
    my_torch::Network net = my_torch::Network::load(o.network);
    const Encoding &encoding = encodingOf(net);
    // Read and check the whole file before printing the first prediction.
    const std::vector<Sample> samples = parseFile(o.chessboards, LabelPolicy::Optional);
    const auto &labels = net.info().labels;

    for (std::size_t begin = 0; begin < samples.size(); begin += PREDICT_CHUNK) {
        const std::size_t end = std::min(begin + PREDICT_CHUNK, samples.size());
        const my_torch::Matrix rows = encodeRows(encoding, samples, begin, end);
        for (const std::size_t c : my_torch::classesOf(net.predict(rows)))
            std::cout << labels[c] << '\n';
    }
    return 0;
}

int train(const Options &o)
{
    my_torch::Network net = my_torch::Network::load(o.network);
    const Encoding &encoding = encodingOf(net);
    const my_torch::TrainingSpec &spec = net.info().training;
    const auto &labels = net.info().labels;
    const std::vector<Sample> samples = parseFile(o.chessboards, LabelPolicy::Required);
    if (samples.empty())
        throw Error(o.chessboards + ": no chessboard to train on");

    // The class of each position; positions are encoded batch by batch during training.
    std::vector<std::size_t> classes(samples.size());
    for (std::size_t r = 0; r < samples.size(); r++) {
        try {
            classes[r] = labelIndex(std::string(labelName(*samples[r].label)), labels);
        } catch (const Error &error) {
            throw Error(o.chessboards + ": position " + std::to_string(r + 1) + ": " +
                        error.what());
        }
    }

    // Validation: the last validation_tail lines in order (ADR-0022), or a seeded fraction drawn
    // at random (ADR-0014).
    const std::size_t tail = spec.config.validationTail;
    std::vector<std::size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::size_t held = tail;
    if (tail == 0) {
        order = my_torch::permutation(samples.size(), spec.config.seed);
        held = static_cast<std::size_t>(spec.config.validation *
                                        static_cast<my_torch::Scalar>(samples.size()));
    }
    if (spec.config.patience > 0 && held == 0)
        throw Error("early stopping (patience) needs a validation set: set validation or "
                    "validation_tail");
    if (held >= samples.size())
        throw Error("the validation set leaves no position to train on");
    // The validation rows are the first `held` of the shuffled order, or the last `tail` lines.
    const auto split = tail ? order.end() - static_cast<std::ptrdiff_t>(held)
                            : order.begin() + static_cast<std::ptrdiff_t>(held);
    const PositionSource validationSet(samples, classes, labels.size(), encoding,
                                       tail ? std::vector<std::size_t>(split, order.end())
                                            : std::vector<std::size_t>(order.begin(), split));
    const PositionSource trainSet(samples, classes, labels.size(), encoding,
                                  tail ? std::vector<std::size_t>(order.begin(), split)
                                       : std::vector<std::size_t>(split, order.end()));

    const auto loss = my_torch::makeLoss(spec.loss);
    const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
    const auto scheduler = my_torch::makeScheduler(spec.scheduler);
    // Progress every N epochs (on stderr, so that the training log on stdout keeps its format),
    // with a snapshot of the weights: the parameters then hold the weights of that epoch.
    my_torch::TrainConfig config = spec.config;
    if (o.progressEvery != 0)
        config.onEpoch = [&](std::size_t epoch, const my_torch::EpochStats &s) {
            const std::size_t finished = epoch + 1;
            if (finished % o.progressEvery != 0)
                return;
            // The time of the save is in the name, so that no snapshot replaces another, even
            // from an earlier run.
            char stamp[32];
            const std::time_t now = std::time(nullptr);
            std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
            const std::string snapshot = (o.save ? *o.save : o.network) + "." + stamp + ".epoch" +
                                         std::to_string(finished) + ".nn";
            net.save(snapshot);
            std::cerr << std::fixed << std::setprecision(4) << "epoch " << finished << "/"
                      << config.epochs << "  loss " << s.trainLoss << "  accuracy "
                      << s.trainAccuracy << "  validation loss " << s.validationLoss
                      << "  validation accuracy " << s.validationAccuracy << "  saved " << snapshot
                      << std::endl;
        };
    const my_torch::History history =
        net.fit(trainSet, held ? &validationSet : nullptr, *loss, *optimizer, config,
                scheduler.get(), o.checkpoint.value_or(""));

    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t e = 0; e < history.size(); e++) {
        const my_torch::EpochStats &s = history[e];
        std::cout << "epoch " << e + 1 << "/" << spec.config.epochs << "  loss " << s.trainLoss
                  << "  accuracy " << s.trainAccuracy;
        if (held)
            std::cout << "  validation loss " << s.validationLoss << "  validation accuracy "
                      << s.validationAccuracy;
        std::cout << '\n';
    }
    net.save(o.save ? *o.save : o.network);
    return 0;
}

} // namespace

std::size_t labelIndex(const std::string &expected, const std::vector<std::string> &labels)
{
    for (std::size_t i = 0; i < labels.size(); i++)
        if (labels[i] == expected)
            return i;
    const std::string firstWord = expected.substr(0, expected.find(' '));
    for (std::size_t i = 0; i < labels.size(); i++)
        if (labels[i] == firstWord)
            return i;
    std::string known;
    for (const std::string &l : labels)
        known += (known.empty() ? "'" : ", '") + l + "'";
    throw Error("expected output '" + expected + "' matches none of the network's labels (" +
                known + ")");
}

int run(const std::vector<std::string> &args)
{
    if (args.size() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        std::cout << usage;
        return 0;
    }
    Options options;
    try {
        options = parseArguments(args);
    } catch (const Error &error) {
        throw Error(std::string(error.what()) + "\n" + usage);
    }
    return options.train ? train(options) : predict(options);
}

} // namespace analyzer
