#include <criterion/criterion.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"
#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: every file name carries the process id.
std::string path(const std::string &base)
{
    return "test_checkpoint_" + std::to_string(getpid()) + "_" + base;
}

struct Interrupted {};

// The training data, failing on purpose once `limit` batches have been read: an interruption in
// the middle of an epoch, as a crash or Ctrl+C would leave it.
class InterruptedSource : public DataSource {
  public:
    InterruptedSource(const Dataset &data, std::size_t limit) : _source(data), _limit(limit) {}

    std::size_t size() const override
    {
        return _source.size();
    }
    std::size_t inputWidth() const override
    {
        return _source.inputWidth();
    }
    std::size_t targetWidth() const override
    {
        return _source.targetWidth();
    }
    void fill(const std::vector<std::size_t> &rows, Matrix &inputs, Matrix &targets) const override
    {
        if (++_batches > _limit)
            throw Interrupted{};
        _source.fill(rows, inputs, targets);
    }

  private:
    MatrixSource _source;
    std::size_t _limit;
    mutable std::size_t _batches = 0;
};

Dataset data(std::uint32_t seed)
{
    Matrix x = gradient_check::random(40, 4, seed);
    Matrix t(40, 3);
    for (std::size_t i = 0; i < 40; i++)
        t(i, (i * 7 + seed) % 3) = 1;
    return {x, t};
}

// Every layer with a state: dense weights and a dropout generator.
Network network()
{
    Network net;
    net.add(std::make_unique<Dense>(4, 8, 1));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dropout>(0.3, 9));
    net.add(std::make_unique<Dense>(8, 3, 2));
    net.add(std::make_unique<Softmax>());
    return net;
}

TrainConfig settings(std::size_t epochs, std::size_t patience = 0)
{
    TrainConfig c;
    c.epochs = epochs;
    c.batchSize = 8; // 5 batches per epoch
    c.patience = patience;
    return c;
}

std::vector<Scalar> weights(Network &net)
{
    std::vector<Scalar> all;
    for (const Parameter &p : net.parameters())
        all.insert(all.end(), p.value->data(), p.value->data() + p.value->size());
    return all;
}

using OptimizerFactory = std::function<std::unique_ptr<Optimizer>()>;

struct Run {
    std::vector<Scalar> weights;
    History history;
};

Run continuous(const OptimizerFactory &make, const TrainConfig &config, const Scheduler *scheduler)
{
    Network net = network();
    const auto optimizer = make();
    CrossEntropy loss;
    const Dataset train = data(3);
    const Dataset validation = data(4);
    const History h = net.fit(train, config.patience ? &validation : nullptr, loss, *optimizer,
                              config, scheduler);
    return {weights(net), h};
}

// Interrupted after `batches` batches, then resumed from the checkpoint with new objects.
Run resumed(const OptimizerFactory &make, const TrainConfig &config, const Scheduler *scheduler,
            std::size_t batches, const std::string &file)
{
    CrossEntropy loss;
    const Dataset train = data(3);
    const Dataset validation = data(4);
    const MatrixSource validationSource(validation);
    {
        Network net = network();
        const auto optimizer = make();
        const InterruptedSource interrupted(train, batches);
        bool stopped = false;
        try {
            net.fit(interrupted, config.patience ? &validationSource : nullptr, loss, *optimizer,
                    config, scheduler, file);
        } catch (const Interrupted &) {
            stopped = true;
        }
        cr_assert(stopped, "the training was not interrupted");
    }
    Network net = network();
    const auto optimizer = make();
    const MatrixSource trainSource(train);
    const History h = net.fit(trainSource, config.patience ? &validationSource : nullptr, loss,
                              *optimizer, config, scheduler, file);
    return {weights(net), h};
}

void assertSame(const Run &a, const Run &b, const char *what)
{
    cr_assert_eq(a.weights.size(), b.weights.size());
    for (std::size_t i = 0; i < a.weights.size(); i++)
        cr_assert_eq(a.weights[i], b.weights[i], "%s: weight %zu: %.17g vs %.17g", what, i,
                     a.weights[i], b.weights[i]);
    cr_assert_eq(a.history.size(), b.history.size(), "%s: %zu vs %zu epochs", what,
                 a.history.size(), b.history.size());
    for (std::size_t e = 0; e < a.history.size(); e++) {
        cr_assert_eq(a.history[e].trainLoss, b.history[e].trainLoss, "%s: epoch %zu", what, e);
        cr_assert_eq(a.history[e].validationLoss, b.history[e].validationLoss, "%s: epoch %zu",
                     what, e);
    }
}

std::string read(const std::string &file)
{
    std::ifstream in(file, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

Test(checkpoint, resumed_training_equals_continuous_training_for_every_optimizer)
{
    const std::vector<std::pair<const char *, OptimizerFactory>> optimizers = {
        {"sgd", [] { return std::make_unique<SGD>(0.1, 0.9, true); }},
        {"adam", [] { return std::make_unique<Adam>(0.01, 0.9, 0.999, 1e-8, true); }},
        {"adamw", [] { return std::make_unique<AdamW>(0.01, 0.1); }},
        {"rmsprop", [] { return std::make_unique<RMSProp>(0.01); }},
        {"adagrad", [] { return std::make_unique<Adagrad>(0.1); }},
    };
    const StepDecay scheduler(0.5, 2, 0.5);

    for (const auto &[name, make] : optimizers) {
        const std::string file = path(name);
        // 17 batches: 3 epochs saved, interrupted in the middle of the 4th.
        const Run a = continuous(make, settings(6), &scheduler);
        const Run b = resumed(make, settings(6), &scheduler, 17, file);
        std::remove(file.c_str());
        assertSame(a, b, name);
    }
}

Test(checkpoint, resumed_training_keeps_the_early_stopping_state)
{
    const OptimizerFactory make = [] { return std::make_unique<Adam>(0.05); };
    const std::string file = path("early");

    const Run a = continuous(make, settings(40, 2), nullptr);
    const std::size_t epochs = a.history.size();
    cr_assert(epochs >= 3 && epochs < 40, "the case must stop early, after %zu epochs", epochs);
    // Interrupted in the 2nd epoch, and in the last one, after the epoch that set the best loss.
    for (std::size_t batches : {std::size_t{7}, 5 * (epochs - 1) + 2}) {
        const Run b = resumed(make, settings(40, 2), nullptr, batches, file);
        std::remove(file.c_str());
        assertSame(a, b, "early stopping");
    }
}

Test(checkpoint, a_finished_training_is_not_run_again_and_can_be_extended)
{
    const OptimizerFactory make = [] { return std::make_unique<Adam>(0.01); };
    const std::string file = path("finished");
    CrossEntropy loss;
    const Dataset train = data(3);

    Network first = network();
    first.fit(train, nullptr, loss, *make(), settings(3), nullptr, file);
    Network again = network();
    const History h = again.fit(train, nullptr, loss, *make(), settings(3), nullptr, file);
    cr_assert_eq(h.size(), 3);
    cr_assert(weights(again) == weights(first));

    // More epochs: training continues from the 3 already run, as one training of 5 would.
    Network extended = network();
    extended.fit(train, nullptr, loss, *make(), settings(5), nullptr, file);
    std::remove(file.c_str());
    const Run five = continuous(make, settings(5), nullptr);
    cr_assert(weights(extended) == five.weights);
}

Test(checkpoint, no_temporary_file_is_left)
{
    const std::string file = path("tmp");
    CrossEntropy loss;
    Adam adam(0.01);
    Network net = network();

    net.fit(data(3), nullptr, loss, adam, settings(2), nullptr, file);
    cr_assert(std::filesystem::exists(file));
    cr_assert(!std::filesystem::exists(file + ".tmp"));
    std::remove(file.c_str());
}

Test(checkpoint, a_corrupted_checkpoint_is_refused)
{
    const std::string file = path("corrupted");
    CrossEntropy loss;
    Adam adam(0.01);
    Network net = network();
    net.fit(data(3), nullptr, loss, adam, settings(2), nullptr, file);

    std::string text = read(file);
    const std::size_t at = text.find("weights\n") + 12;
    text[at] = text[at] == '1' ? '2' : '1';
    std::ofstream(file, std::ios::binary) << text;
    Network other = network();
    try {
        other.fit(data(3), nullptr, loss, adam, settings(4), nullptr, file);
        cr_assert_fail("a corrupted checkpoint was accepted");
    } catch (const Error &e) {
        cr_assert(std::string(e.what()).find("checksum mismatch") != std::string::npos, "got '%s'",
                  e.what());
    }

    std::ofstream(file, std::ios::binary) << text.substr(0, text.size() / 2);
    try {
        other.fit(data(3), nullptr, loss, adam, settings(4), nullptr, file);
        cr_assert_fail("a truncated checkpoint was accepted");
    } catch (const Error &e) {
        cr_assert(std::string(e.what()).find("truncated") != std::string::npos, "got '%s'",
                  e.what());
    }
    std::remove(file.c_str());
}

Test(checkpoint, a_checkpoint_of_another_training_is_refused)
{
    const std::string file = path("other");
    CrossEntropy loss;
    Adam adam(0.01);
    Network net = network();
    net.fit(data(3), nullptr, loss, adam, settings(2), nullptr, file);

    TrainConfig otherBatch = settings(4);
    otherBatch.batchSize = 4;
    SGD sgd(0.1);
    Network wider;
    wider.add(std::make_unique<Dense>(4, 9, 1));
    wider.add(std::make_unique<Dense>(9, 3, 2));
    wider.add(std::make_unique<Softmax>());
    const std::vector<std::function<void()>> others = {
        [&] { network().fit(data(3), nullptr, loss, adam, otherBatch, nullptr, file); },
        [&] { network().fit(data(3), nullptr, loss, sgd, settings(4), nullptr, file); },
        [&] { wider.fit(data(3), nullptr, loss, adam, settings(4), nullptr, file); },
        [&] {
            const Dataset fewer{Matrix(20, 4), Matrix(20, 3)};
            network().fit(fewer, nullptr, loss, adam, settings(4), nullptr, file);
        },
    };
    for (std::size_t i = 0; i < others.size(); i++) {
        try {
            others[i]();
            cr_assert_fail("case %zu: the checkpoint of another training was accepted", i);
        } catch (const Error &e) {
            cr_assert(std::string(e.what()).find("another training") != std::string::npos,
                      "case %zu: got '%s'", i, e.what());
        }
    }
    std::remove(file.c_str());
}

Test(checkpoint, matrices_read_back_exactly)
{
    Matrix m(2, 3);
    const Scalar values[] = {0.1, -1.0 / 3, 1e-300, 123456789.123456789, -0.5, 2.0 / 7};
    for (std::size_t i = 0; i < 6; i++)
        m.data()[i] = values[i];
    std::stringstream text;

    writeMatrices(text, {m, Matrix(0, 0)});
    const std::vector<Matrix> back = readMatrices(text);
    cr_assert_eq(back.size(), 2);
    for (std::size_t i = 0; i < 6; i++)
        cr_assert_eq(back[0].data()[i], values[i]);
    cr_assert_eq(back[1].size(), 0);

    std::stringstream bad("1\n2 2\n1 2 3\n");
    cr_assert_throw(readMatrices(bad), Error);
}
