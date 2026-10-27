// Shows overfitting and what each remedy of #7 does against it (docs/performance/08-overfitting.md).
// Two interleaved spirals with few noisy training points and a network much larger than needed:
// it can memorize the training set. Three sets per seed: train (80 points), validation (500 points,
// used to pick the regularization strength and the early-stopping epoch) and test (500 points,
// never used for a choice, reported). 2-64-64-2 network, tanh, Softmax + cross-entropy, Adam 0.01,
// batches of 10, 800 epochs, 5 seeds; medians over seeds.
//
// Usage: bench_overfitting <output directory>   (see scripts/bench_overfitting.sh)
// Writes: overfitting_grid.csv (every configuration, final medians), overfitting_curves.csv (median
// curves of the baseline and of the best configuration of each remedy).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

constexpr std::size_t epochs = 800;
constexpr std::uint32_t seeds = 5;

Dataset spirals(std::size_t perClass, Scalar noiseLevel, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<Scalar> noise(0, noiseLevel);
    std::uniform_real_distribution<Scalar> position(0.05, 1.0);
    Dataset d{Matrix(2 * perClass, 2), Matrix(2 * perClass, 2)};

    for (std::size_t i = 0; i < perClass; i++)
        for (std::size_t cls = 0; cls < 2; cls++) {
            const Scalar radius = position(rng);
            const Scalar angle = 2 * std::numbers::pi * radius + cls * std::numbers::pi;
            const std::size_t row = 2 * i + cls;
            d.inputs(row, 0) = radius * std::cos(angle) + noise(rng);
            d.inputs(row, 1) = radius * std::sin(angle) + noise(rng);
            d.targets(row, cls) = 1;
        }
    return d;
}

struct Setting {
    std::string name;
    std::string family; // baseline, l1, l2, elastic_net, dropout, early_stopping
    Scalar l1 = 0;
    Scalar l2 = 0;
    Scalar dropout = 0;
    std::size_t patience = 0;
};

struct Curve {
    std::vector<Scalar> train;
    std::vector<Scalar> validation;
    Scalar testLoss = 0;
    Scalar testAccuracy = 0;
    Scalar validationLoss = 0;
    std::size_t epochsRun = 0;
};

Scalar accuracy(const Matrix &p, const Matrix &t)
{
    std::size_t right = 0;
    for (std::size_t r = 0; r < p.rows(); r++)
        right += (p(r, 1) > p(r, 0)) == (t(r, 1) == 1);
    return static_cast<Scalar>(right) / static_cast<Scalar>(p.rows());
}

Curve run(const Setting &s, std::uint32_t seed)
{
    const Dataset train = spirals(40, 0.1, 1000 + seed);
    const Dataset validation = spirals(250, 0.1, 2000 + seed);
    const Dataset test = spirals(250, 0.1, 3000 + seed);
    Network net;
    const Regularization reg{s.l1, s.l2};

    net.add(std::make_unique<Dense>(2, 64, 10 * seed + 1, reg));
    net.add(std::make_unique<Tanh>());
    if (s.dropout > 0)
        net.add(std::make_unique<Dropout>(s.dropout, 10 * seed + 2));
    net.add(std::make_unique<Dense>(64, 64, 10 * seed + 3, reg));
    net.add(std::make_unique<Tanh>());
    if (s.dropout > 0)
        net.add(std::make_unique<Dropout>(s.dropout, 10 * seed + 4));
    net.add(std::make_unique<Dense>(64, 2, 10 * seed + 5, reg));
    net.add(std::make_unique<Softmax>());
    CrossEntropy loss;
    Adam adam(0.01);
    TrainConfig config;
    config.epochs = epochs;
    config.batchSize = 10;
    config.seed = seed;
    config.patience = s.patience;

    const History h = net.fit(train, &validation, loss, adam, config);
    Curve c;
    for (const EpochStats &e : h) {
        c.train.push_back(e.trainLoss);
        c.validation.push_back(e.validationLoss);
    }
    const Matrix p = net.predict(test.inputs);
    c.testLoss = loss.value(p, test.targets);
    c.testAccuracy = accuracy(p, test.targets);
    c.validationLoss = loss.value(net.predict(validation.inputs), validation.targets);
    c.epochsRun = h.size();
    return c;
}

Scalar median(std::vector<Scalar> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Median curve over seeds; a run stopped early keeps its last value afterwards.
std::vector<Scalar> medianCurve(const std::vector<Curve> &runs, bool validation)
{
    std::vector<Scalar> out;
    for (std::size_t e = 0; e < epochs; e++) {
        std::vector<Scalar> values;
        for (const Curve &c : runs) {
            const auto &v = validation ? c.validation : c.train;
            values.push_back(v[std::min(e, v.size() - 1)]);
        }
        out.push_back(median(values));
    }
    return out;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <output directory>\n";
        return 84;
    }
    const std::string dir = argv[1];
    std::vector<Setting> settings = {{"none", "baseline"}};
    const auto label = [](Scalar v) {
        std::ostringstream out;
        out << v;
        return out.str();
    };
    for (Scalar l1 : {1e-5, 3e-5, 1e-4, 3e-4, 1e-3})
        settings.push_back({"l1=" + label(l1), "l1", l1});
    for (Scalar l2 : {1e-4, 3e-4, 1e-3, 3e-3, 1e-2})
        settings.push_back({"l2=" + label(l2), "l2", 0, l2});
    // Elastic net: the same coefficient for L1 and L2.
    for (Scalar both : {3e-5, 1e-4, 3e-4, 1e-3})
        settings.push_back({"elastic=" + label(both), "elastic_net", both, both});
    for (Scalar rate : {0.1, 0.2, 0.3, 0.5})
        settings.push_back({"dropout=" + label(rate), "dropout", 0, 0, rate});
    for (std::size_t patience : {20, 50})
        settings.push_back(
            {"patience=" + std::to_string(patience), "early_stopping", 0, 0, 0, patience});

    std::vector<std::vector<Curve>> results(settings.size());
    for (std::size_t i = 0; i < settings.size(); i++) {
        std::vector<std::future<Curve>> jobs;
        for (std::uint32_t seed = 0; seed < seeds; seed++)
            jobs.push_back(std::async(std::launch::async, run, settings[i], seed));
        for (auto &job : jobs)
            results[i].push_back(job.get());
        std::cerr << settings[i].name << " done\n";
    }

    std::ofstream grid(dir + "/overfitting_grid.csv");
    grid << "setting,family,validation_loss,test_loss,test_loss_min,test_loss_max,test_accuracy,"
            "epochs_run\n";
    const std::vector<std::string> families = {"baseline",    "l1",      "l2",
                                               "elastic_net", "dropout", "early_stopping"};
    std::vector<std::size_t> bestOf(families.size(), settings.size());
    for (std::size_t i = 0; i < settings.size(); i++) {
        std::vector<Scalar> val, test, acc, ran;
        for (const Curve &c : results[i]) {
            val.push_back(c.validationLoss);
            test.push_back(c.testLoss);
            acc.push_back(c.testAccuracy);
            ran.push_back(static_cast<Scalar>(c.epochsRun));
        }
        grid << settings[i].name << ',' << settings[i].family << ',' << median(val) << ','
             << median(test) << ',' << *std::min_element(test.begin(), test.end()) << ','
             << *std::max_element(test.begin(), test.end()) << ',' << median(acc) << ','
             << median(ran) << '\n';
        // The best setting of each family is chosen on the validation loss, never on the test.
        const std::size_t f = std::find(families.begin(), families.end(), settings[i].family) -
                              families.begin();
        std::vector<Scalar> bestVal;
        if (bestOf[f] < settings.size())
            for (const Curve &c : results[bestOf[f]])
                bestVal.push_back(c.validationLoss);
        if (bestOf[f] == settings.size() || median(val) < median(bestVal))
            bestOf[f] = i;
    }

    std::ofstream curves(dir + "/overfitting_curves.csv");
    curves << "setting,epoch,train_loss,validation_loss\n";
    for (std::size_t i : bestOf) {
        const auto train = medianCurve(results[i], false);
        const auto validation = medianCurve(results[i], true);
        for (std::size_t e = 0; e < epochs; e++)
            curves << settings[i].name << ',' << e << ',' << train[e] << ',' << validation[e]
                   << '\n';
    }
    return 0;
}
