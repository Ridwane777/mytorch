#include "my_torch/Network.hpp"

#include <limits>
#include <numeric>
#include <random>

#include "my_torch/Error.hpp"
#include "my_torch/Metrics.hpp"

namespace my_torch {

namespace {

// Uniform integer in [0, range) by rejection sampling. std::uniform_int_distribution and
// std::shuffle are implementation-defined, so the same seed could shuffle differently with another
// standard library; std::mt19937_64 itself is fully specified (ADR-0009).
std::size_t uniformBelow(std::mt19937_64 &rng, std::size_t range)
{
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t limit = max - max % range;
    std::uint64_t draw = rng();

    while (draw >= limit)
        draw = rng();
    return static_cast<std::size_t>(draw % range);
}

// Fisher-Yates shuffle.
void shuffle(std::vector<std::size_t> &order, std::mt19937_64 &rng)
{
    for (std::size_t i = order.size(); i > 1; i--)
        std::swap(order[i - 1], order[uniformBelow(rng, i)]);
}

Matrix gatherRows(const Matrix &source, const std::vector<std::size_t> &order, std::size_t begin,
                  std::size_t end)
{
    Matrix batch(end - begin, source.cols());

    for (std::size_t r = begin; r < end; r++)
        for (std::size_t c = 0; c < source.cols(); c++)
            batch(r - begin, c) = source(order[r], c);
    return batch;
}

// Rows whose predicted class matches the target's (ADR-0011).
std::size_t countCorrect(const Matrix &prediction, const Matrix &targets)
{
    const auto predicted = classesOf(prediction);
    const auto expected = classesOf(targets);
    std::size_t correct = 0;

    for (std::size_t r = 0; r < predicted.size(); r++)
        correct += predicted[r] == expected[r];
    return correct;
}

void checkDataset(const Dataset &data, const char *name)
{
    if (data.inputs.rows() == 0)
        throw Error(std::string("fit: ") + name + " set is empty");
    if (data.inputs.rows() != data.targets.rows())
        throw Error(std::string("fit: ") + name + " set has " +
                    std::to_string(data.inputs.rows()) + " inputs but " +
                    std::to_string(data.targets.rows()) + " targets");
}

} // namespace

void Network::add(std::unique_ptr<Layer> layer)
{
    if (!layer)
        throw Error("add: null layer");
    _layers.push_back(std::move(layer));
}

Matrix Network::forward(const Matrix &input, bool training)
{
    Matrix output = input;

    for (auto &layer : _layers)
        output = layer->forward(output, training);
    return output;
}

Matrix Network::predict(const Matrix &input)
{
    return forward(input, false);
}

std::vector<Parameter> Network::parameters()
{
    std::vector<Parameter> all;

    for (auto &layer : _layers)
        for (const Parameter &p : layer->parameters())
            all.push_back(p);
    return all;
}

const std::vector<std::unique_ptr<Layer>> &Network::layers() const noexcept
{
    return _layers;
}

History Network::fit(const Dataset &train, const Dataset *validation, Loss &loss,
                     Optimizer &optimizer, const TrainConfig &config, const Scheduler *scheduler)
{
    checkDataset(train, "training");
    if (validation)
        checkDataset(*validation, "validation");
    if (config.batchSize == 0)
        throw Error("fit: batch size must be positive");
    if (_layers.empty())
        throw Error("fit: the network has no layer");
    if (config.patience > 0 && !validation)
        throw Error("fit: early stopping needs a validation set");
    const bool skipSoftmax = loss.fusesSoftmax();
    if (skipSoftmax && _layers.back()->type() != "softmax")
        throw Error("fit: " + loss.type() + " needs a final softmax layer");

    const std::size_t count = train.inputs.rows();
    std::vector<std::size_t> order(count);
    std::mt19937_64 rng(config.seed);
    const std::vector<Parameter> params = parameters();
    const std::size_t lastBackward = skipSoftmax ? _layers.size() - 1 : _layers.size();
    History history;
    Scalar best = 0;
    std::size_t sinceBest = 0;
    std::vector<Matrix> bestWeights;

    for (std::size_t epoch = 0; epoch < config.epochs; epoch++) {
        if (scheduler)
            optimizer.setLearningRate(scheduler->rate(epoch));
        std::iota(order.begin(), order.end(), 0);
        if (config.shuffle)
            shuffle(order, rng);
        Scalar lossSum = 0;
        Scalar penaltySum = 0;
        std::size_t batches = 0;
        std::size_t correct = 0;
        for (std::size_t begin = 0; begin < count; begin += config.batchSize) {
            const std::size_t end = std::min(begin + config.batchSize, count);
            const Matrix inputs = gatherRows(train.inputs, order, begin, end);
            const Matrix targets = gatherRows(train.targets, order, begin, end);

            for (const Parameter &p : params)
                p.grad->fill(0);
            const Matrix prediction = forward(inputs, true);
            // The loss is a batch mean: weight it by the batch size so that a smaller last batch
            // does not count as much as a full one.
            lossSum += loss.value(prediction, targets) * static_cast<Scalar>(end - begin);
            for (const auto &layer : _layers)
                penaltySum += layer->penalty();
            batches++;
            correct += countCorrect(prediction, targets);
            Matrix grad = loss.gradient(prediction, targets);
            for (std::size_t i = lastBackward; i-- > 0;)
                grad = _layers[i]->backward(grad);
            optimizer.step(params);
        }
        EpochStats stats;
        stats.trainLoss = lossSum / static_cast<Scalar>(count);
        stats.penalty = penaltySum / static_cast<Scalar>(batches);
        stats.trainAccuracy = static_cast<Scalar>(correct) / static_cast<Scalar>(count);
        if (validation) {
            const Matrix predicted = predict(validation->inputs);
            stats.validationLoss = loss.value(predicted, validation->targets);
            stats.validationAccuracy =
                static_cast<Scalar>(countCorrect(predicted, validation->targets)) /
                static_cast<Scalar>(predicted.rows());
        }
        history.push_back(stats);
        if (config.patience == 0)
            continue;
        if (bestWeights.empty() || stats.validationLoss < best - config.minDelta) {
            best = stats.validationLoss;
            sinceBest = 0;
            bestWeights.clear();
            for (const Parameter &p : params)
                bestWeights.push_back(*p.value);
        } else if (++sinceBest >= config.patience) {
            break;
        }
    }
    if (config.restoreBest && !bestWeights.empty())
        for (std::size_t i = 0; i < params.size(); i++)
            *params[i].value = bestWeights[i];
    return history;
}

} // namespace my_torch
