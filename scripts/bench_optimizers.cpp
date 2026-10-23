// Compares my_torch's optimizers and schedulers on a non-linear problem: two interleaved spirals
// to tell apart (2 classes, 200 points). 2-32-32-1 network (tanh, sigmoid), squared-error loss,
// full batch, fixed seeds, so the result is reproducible.
//
// Every configuration is trained on SEEDS different (data noise, weight initialisation) pairs and
// the median over seeds is reported, with the min-max range for the final loss.
//
// Usage: bench_optimizers <output directory>   (see scripts/bench_optimizers.sh)
//
// Writes: optimizer_grid.csv (every learning rate tried), optimizers.csv (median curve at each
// optimizer's best rate), schedulers.csv (median curve of Adam under each scheduler).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

constexpr std::size_t epochs = 400;
constexpr std::uint32_t seeds = 5;

struct Problem {
    Matrix x;
    Matrix y;
};

Problem spirals(std::size_t perClass, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<Scalar> noise(0, 0.05);
    Problem p{Matrix(2 * perClass, 2), Matrix(2 * perClass, 1)};

    for (std::size_t i = 0; i < perClass; i++) {
        // Start slightly away from the origin, where the two classes would overlap.
        const Scalar radius = 0.05 + 0.95 * static_cast<Scalar>(i) / perClass;
        for (std::size_t cls = 0; cls < 2; cls++) {
            const Scalar angle = 4 * std::numbers::pi * radius + cls * std::numbers::pi;
            const std::size_t row = 2 * i + cls;
            p.x(row, 0) = radius * std::cos(angle) + noise(rng);
            p.x(row, 1) = radius * std::sin(angle) + noise(rng);
            p.y(row, 0) = static_cast<Scalar>(cls);
        }
    }
    return p;
}

struct Point {
    Scalar loss;
    Scalar accuracy;
    Scalar rate;
};

// Trains a fresh network. When `scheduler` is set, it sets the rate at the start of each epoch.
std::vector<Point> train(const Problem &problem, std::uint32_t seed, Optimizer &optimizer,
                         const Scheduler *scheduler)
{
    Dense first(2, 32, 3 * seed + 1);
    Dense second(32, 32, 3 * seed + 2);
    Dense third(32, 1, 3 * seed + 3);
    Tanh a;
    Tanh b;
    Sigmoid out;
    std::vector<Parameter> params;
    std::vector<Point> curve;
    const Scalar n = static_cast<Scalar>(problem.x.rows());

    for (Layer *layer : std::initializer_list<Layer *>{&first, &second, &third})
        for (const Parameter &p : layer->parameters())
            params.push_back(p);
    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        if (scheduler)
            optimizer.setLearningRate(scheduler->rate(epoch));
        for (const Parameter &p : params)
            p.grad->fill(0);
        const Matrix p = out.forward(
            third.forward(
                b.forward(second.forward(a.forward(first.forward(problem.x, true), true), true),
                          true),
                true),
            true);
        Matrix grad(p.rows(), 1);
        Scalar loss = 0;
        int correct = 0;

        for (std::size_t i = 0; i < p.rows(); i++) {
            const Scalar d = p(i, 0) - problem.y(i, 0);
            loss += d * d / n;
            grad(i, 0) = 2 * d / n;
            correct += (p(i, 0) > 0.5) == (problem.y(i, 0) > 0.5);
        }
        // A diverged run gives NaN: count it as 1, a loss no useful run reaches.
        curve.push_back({std::isfinite(loss) ? loss : 1.0, correct / n, optimizer.learningRate()});
        first.backward(a.backward(second.backward(b.backward(third.backward(out.backward(grad))))));
        optimizer.step(params);
    }
    return curve;
}

Scalar median(std::vector<Scalar> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Runs the seeds in parallel; make(seed) builds the optimizer, scheduler may be null.
using Curves = std::vector<std::vector<Point>>;

Curves trainAllSeeds(const std::vector<Problem> &problems,
                     const std::function<std::unique_ptr<Optimizer>()> &makeOptimizer,
                     const Scheduler *scheduler)
{
    std::vector<std::future<std::vector<Point>>> jobs;
    Curves curves;

    for (std::uint32_t s = 0; s < seeds; s++)
        jobs.push_back(std::async(std::launch::async, [&, s] {
            auto optimizer = makeOptimizer();
            return train(problems[s], s, *optimizer, scheduler);
        }));
    for (auto &job : jobs)
        curves.push_back(job.get());
    return curves;
}

// Median over seeds of one field, at one epoch.
Scalar medianAt(const Curves &curves, std::size_t epoch, Scalar Point::*field)
{
    std::vector<Scalar> values;

    for (const auto &curve : curves)
        values.push_back(curve[epoch].*field);
    return median(values);
}

struct Candidate {
    std::string name;
    std::function<std::unique_ptr<Optimizer>(Scalar)> make;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: bench_optimizers <output directory>\n";
        return 84;
    }
    const std::string dir = argv[1];
    std::vector<Problem> problems;
    for (std::uint32_t s = 0; s < seeds; s++)
        problems.push_back(spirals(100, 7 + s));
    const std::vector<Scalar> rates = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0};
    const std::vector<Candidate> candidates = {
        {"SGD", [](Scalar r) { return std::make_unique<SGD>(r); }},
        {"SGD+momentum", [](Scalar r) { return std::make_unique<SGD>(r, 0.9); }},
        {"SGD+Nesterov", [](Scalar r) { return std::make_unique<SGD>(r, 0.9, true); }},
        {"Adam", [](Scalar r) { return std::make_unique<Adam>(r); }},
        {"AdamW", [](Scalar r) { return std::make_unique<AdamW>(r, 0.01); }},
        {"RMSProp", [](Scalar r) { return std::make_unique<RMSProp>(r); }},
        {"Adagrad", [](Scalar r) { return std::make_unique<Adagrad>(r); }},
    };
    std::ofstream grid(dir + "/optimizer_grid.csv");
    std::ofstream best(dir + "/optimizers.csv");
    Scalar adamRate = 0.01;

    grid << "optimizer,learning_rate,median_final_loss,min_final_loss,max_final_loss,"
            "median_final_accuracy\n";
    best << "optimizer,learning_rate,epoch,loss,accuracy\n";
    for (const Candidate &candidate : candidates) {
        Scalar bestLoss = 1e300;
        Scalar bestRate = rates[0];
        Curves bestCurves;

        for (Scalar rate : rates) {
            const Curves curves =
                trainAllSeeds(problems, [&] { return candidate.make(rate); }, nullptr);
            std::vector<Scalar> finals;
            for (const auto &curve : curves)
                finals.push_back(curve.back().loss);
            const Scalar loss = median(finals);

            grid << candidate.name << ',' << rate << ',' << loss << ','
                 << *std::min_element(finals.begin(), finals.end()) << ','
                 << *std::max_element(finals.begin(), finals.end()) << ','
                 << medianAt(curves, epochs - 1, &Point::accuracy) << '\n';
            // The best rate is chosen on the median final training loss.
            if (loss < bestLoss) {
                bestLoss = loss;
                bestRate = rate;
                bestCurves = curves;
            }
        }
        for (std::size_t e = 0; e < epochs; e++)
            best << candidate.name << ',' << bestRate << ',' << e << ','
                 << medianAt(bestCurves, e, &Point::loss) << ','
                 << medianAt(bestCurves, e, &Point::accuracy) << '\n';
        if (candidate.name == "Adam")
            adamRate = bestRate;
    }

    // Schedulers: Adam starting from the rate that was best at a constant rate.
    std::vector<std::pair<std::string, std::unique_ptr<Scheduler>>> schedulers;
    schedulers.emplace_back("constant", std::make_unique<ConstantRate>(adamRate));
    schedulers.emplace_back("step (/3 every 100 epochs)",
                            std::make_unique<StepDecay>(adamRate, 100, 1.0 / 3));
    schedulers.emplace_back("exponential (0.995)",
                            std::make_unique<ExponentialDecay>(adamRate, 0.995));
    schedulers.emplace_back("cosine", std::make_unique<CosineAnnealing>(adamRate, epochs));
    schedulers.emplace_back(
        "warmup 20 + cosine",
        std::make_unique<Warmup>(std::make_unique<CosineAnnealing>(adamRate, epochs - 20), 20));
    std::ofstream out(dir + "/schedulers.csv");

    out << "scheduler,epoch,learning_rate,loss,accuracy\n";
    for (const auto &[name, scheduler] : schedulers) {
        const Curves curves = trainAllSeeds(
            problems, [&] { return std::make_unique<Adam>(adamRate); }, scheduler.get());

        for (std::size_t e = 0; e < epochs; e++)
            out << name << ',' << e << ',' << medianAt(curves, e, &Point::rate) << ','
                << medianAt(curves, e, &Point::loss) << ',' << medianAt(curves, e, &Point::accuracy)
                << '\n';
    }
    std::cout << "Adam base rate: " << adamRate << '\n';
    return 0;
}
