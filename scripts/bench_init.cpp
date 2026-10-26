// Compares my_torch's weight initialisations (ADR-0020) on a non-linear problem: two interleaved
// spirals to tell apart (2 classes, 200 points, same generator as bench_optimizers). Deeper
// network than the optimizer benchmark, where the initialisation matters more:
// 2-64-64-64-1 (four Dense layers), sigmoid output, squared-error loss, full batch.
//
// Two families, each with the activation its initialisations were designed for:
//   tanh:  uniform (the default), lecun_*, xavier_*, orthogonal
//   relu:  uniform (the default), he_*, xavier_*, orthogonal
// The optimizer is fixed (Adam, learning rate 0.001, no scheduler) and the same for every run:
// only the initialisation changes. Every initialisation is trained on SEEDS different (data
// noise, weight seed) pairs, shared by all initialisations, and the median over seeds is
// reported, with the min-max range.
//
// Usage: bench_init <output directory>   (see scripts/bench_init.sh)
//
// Writes: init_curves.csv (median and min-max loss at each epoch), init_summary.csv (loss
// before the first update, loss after the first epoch's update, final loss).
#include <algorithm>
#include <cmath>
#include <fstream>
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
constexpr Scalar learningRate = 0.001; // Adam, fixed for every run
constexpr std::size_t width = 64;      // hidden width; three hidden layers

struct Problem {
    Matrix x;
    Matrix y;
};

// Same generator as scripts/bench_optimizers.cpp.
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
};

enum class Family { Tanh, ReLU };

const char *familyName(Family f)
{
    return f == Family::Tanh ? "tanh" : "relu";
}

// Trains a fresh network. curve[e].loss is the loss of the forward pass of epoch e, i.e. after e
// updates: curve[0] is the loss of the freshly initialised network.
std::vector<Point> train(const Problem &problem, std::uint32_t seed, Family family, Init init)
{
    std::vector<std::unique_ptr<Layer>> layers;
    const std::size_t sizes[] = {2, width, width, width, 1};

    for (std::size_t i = 0; i < 4; i++) {
        layers.push_back(
            std::make_unique<Dense>(sizes[i], sizes[i + 1], 4 * seed + i, Regularization{}, init));
        if (i == 3)
            layers.push_back(std::make_unique<Sigmoid>());
        else if (family == Family::Tanh)
            layers.push_back(std::make_unique<Tanh>());
        else
            layers.push_back(std::make_unique<ReLU>());
    }
    std::vector<Parameter> params;
    for (const auto &layer : layers)
        for (const Parameter &p : layer->parameters())
            params.push_back(p);
    Adam optimizer(learningRate);
    std::vector<Point> curve;
    const Scalar n = static_cast<Scalar>(problem.x.rows());

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        for (const Parameter &p : params)
            p.grad->fill(0);
        Matrix out = problem.x;
        for (const auto &layer : layers)
            out = layer->forward(out, true);
        Matrix grad(out.rows(), 1);
        Scalar loss = 0;
        int correct = 0;

        for (std::size_t i = 0; i < out.rows(); i++) {
            const Scalar d = out(i, 0) - problem.y(i, 0);
            loss += d * d / n;
            grad(i, 0) = 2 * d / n;
            correct += (out(i, 0) > 0.5) == (problem.y(i, 0) > 0.5);
        }
        // A diverged run gives NaN: count it as 1, a loss no useful run reaches.
        curve.push_back({std::isfinite(loss) ? loss : 1.0, correct / n});
        for (std::size_t i = layers.size(); i-- > 0;)
            grad = layers[i]->backward(grad);
        optimizer.step(params);
    }
    return curve;
}

Scalar median(std::vector<Scalar> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Candidate {
    Family family;
    std::string name;
    Init init;
};

using Curves = std::vector<std::vector<Point>>;

Curves trainAllSeeds(const std::vector<Problem> &problems, const Candidate &candidate)
{
    std::vector<std::future<std::vector<Point>>> jobs;
    Curves curves;

    for (std::uint32_t s = 0; s < seeds; s++)
        jobs.push_back(std::async(std::launch::async, [&, s] {
            return train(problems[s], s, candidate.family, candidate.init);
        }));
    for (auto &job : jobs)
        curves.push_back(job.get());
    return curves;
}

// Median, min and max over seeds of the loss (or accuracy) at one epoch.
struct Spread {
    Scalar median;
    Scalar min;
    Scalar max;
};

Spread spreadAt(const Curves &curves, std::size_t epoch, Scalar Point::*field)
{
    std::vector<Scalar> values;

    for (const auto &curve : curves)
        values.push_back(curve[epoch].*field);
    return {median(values), *std::min_element(values.begin(), values.end()),
            *std::max_element(values.begin(), values.end())};
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: bench_init <output directory>\n";
        return 84;
    }
    const std::string dir = argv[1];
    std::vector<Problem> problems;
    for (std::uint32_t s = 0; s < seeds; s++)
        problems.push_back(spirals(100, 7 + s));
    const std::vector<Candidate> candidates = {
        {Family::Tanh, "uniform", Init::Uniform},
        {Family::Tanh, "lecun_uniform", Init::LecunUniform},
        {Family::Tanh, "lecun_normal", Init::LecunNormal},
        {Family::Tanh, "xavier_uniform", Init::XavierUniform},
        {Family::Tanh, "xavier_normal", Init::XavierNormal},
        {Family::Tanh, "orthogonal", Init::Orthogonal},
        {Family::ReLU, "uniform", Init::Uniform},
        {Family::ReLU, "he_uniform", Init::HeUniform},
        {Family::ReLU, "he_normal", Init::HeNormal},
        {Family::ReLU, "xavier_uniform", Init::XavierUniform},
        {Family::ReLU, "xavier_normal", Init::XavierNormal},
        {Family::ReLU, "orthogonal", Init::Orthogonal},
    };
    std::ofstream curvesOut(dir + "/init_curves.csv");
    std::ofstream summary(dir + "/init_summary.csv");

    curvesOut << "family,init,epoch,median_loss,min_loss,max_loss,median_accuracy\n";
    summary << "family,init,median_first_epoch_loss,median_loss_epoch_10,median_final_loss,"
               "min_final_loss,max_final_loss,median_final_accuracy\n";
    for (const Candidate &candidate : candidates) {
        const Curves curves = trainAllSeeds(problems, candidate);

        for (std::size_t e = 0; e < epochs; e++) {
            const Spread loss = spreadAt(curves, e, &Point::loss);
            curvesOut << familyName(candidate.family) << ',' << candidate.name << ',' << e << ','
                      << loss.median << ',' << loss.min << ',' << loss.max << ','
                      << spreadAt(curves, e, &Point::accuracy).median << '\n';
        }
        const Spread last = spreadAt(curves, epochs - 1, &Point::loss);
        summary << familyName(candidate.family) << ',' << candidate.name << ','
                << spreadAt(curves, 0, &Point::loss).median << ','
                << spreadAt(curves, 10, &Point::loss).median << ',' << last.median << ','
                << last.min << ',' << last.max << ','
                << spreadAt(curves, epochs - 1, &Point::accuracy).median << '\n';
    }
    return 0;
}
