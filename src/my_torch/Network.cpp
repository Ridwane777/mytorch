#include "my_torch/Network.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>

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

void checkSource(const DataSource &data, const char *name)
{
    if (data.size() == 0)
        throw Error(std::string("fit: ") + name + " set is empty");
    if (data.inputWidth() == 0 || data.targetWidth() == 0)
        throw Error(std::string("fit: ") + name + " set has an empty input or target width");
}

// Batch of the given rows, built by the source into zeroed matrices.
// A parameter with the 1-based index and the type of its layer, for the error messages.
struct OwnedParameter {
    Parameter parameter;
    std::size_t layer;
    std::string type;
};

bool allFinite(const Matrix &m)
{
    const Scalar *v = m.data();
    for (std::size_t i = 0; i < m.size(); i++)
        if (!std::isfinite(v[i]))
            return false;
    return true;
}

std::string position(std::size_t epoch, std::size_t batch)
{
    return " at epoch " + std::to_string(epoch + 1) + ", batch " + std::to_string(batch + 1);
}

// Throws on the first parameter whose value (weights) or gradient is not finite.
void checkFinite(const std::vector<OwnedParameter> &params, bool gradients, std::size_t epoch,
                 std::size_t batch)
{
    for (const OwnedParameter &p : params)
        if (!allFinite(gradients ? *p.parameter.grad : *p.parameter.value))
            throw Error(std::string("fit: non-finite ") + (gradients ? "gradient" : "weight") +
                        " in layer " + std::to_string(p.layer) + " (" + p.type + ")" +
                        position(epoch, batch) +
                        "; lower the learning rate or set clip_norm / clip_value");
}

// Gradient clipping (ADR-0031): by value first, then by global norm.
void clip(const std::vector<OwnedParameter> &params, const TrainConfig &config)
{
    if (config.clipValue > 0)
        for (const OwnedParameter &p : params) {
            Scalar *g = p.parameter.grad->data();
            for (std::size_t i = 0; i < p.parameter.grad->size(); i++)
                g[i] = std::clamp(g[i], -config.clipValue, config.clipValue);
        }
    if (config.clipNorm <= 0)
        return;
    Scalar squares = 0;
    for (const OwnedParameter &p : params) {
        const Scalar *g = p.parameter.grad->data();
        for (std::size_t i = 0; i < p.parameter.grad->size(); i++)
            squares += g[i] * g[i];
    }
    const Scalar norm = std::sqrt(squares);
    if (norm <= config.clipNorm)
        return;
    const Scalar scale = config.clipNorm / norm;
    for (const OwnedParameter &p : params) {
        Scalar *g = p.parameter.grad->data();
        for (std::size_t i = 0; i < p.parameter.grad->size(); i++)
            g[i] *= scale;
    }
}

std::pair<Matrix, Matrix> batchOf(const DataSource &data, const std::vector<std::size_t> &rows)
{
    Matrix inputs(rows.size(), data.inputWidth());
    Matrix targets(rows.size(), data.targetWidth());
    data.fill(rows, inputs, targets);
    return {std::move(inputs), std::move(targets)};
}

} // namespace

MatrixSource::MatrixSource(const Dataset &data) : _data(data)
{
    if (data.inputs.rows() != data.targets.rows())
        throw Error("fit: " + std::to_string(data.inputs.rows()) + " inputs but " +
                    std::to_string(data.targets.rows()) + " targets");
}

std::size_t MatrixSource::size() const
{
    return _data.inputs.rows();
}

std::size_t MatrixSource::inputWidth() const
{
    return _data.inputs.cols();
}

std::size_t MatrixSource::targetWidth() const
{
    return _data.targets.cols();
}

void MatrixSource::fill(const std::vector<std::size_t> &rows, Matrix &inputs,
                        Matrix &targets) const
{
    for (std::size_t r = 0; r < rows.size(); r++) {
        for (std::size_t c = 0; c < inputs.cols(); c++)
            inputs(r, c) = _data.inputs(rows[r], c);
        for (std::size_t c = 0; c < targets.cols(); c++)
            targets(r, c) = _data.targets(rows[r], c);
    }
}

std::vector<std::size_t> permutation(std::size_t count, std::uint64_t seed)
{
    std::vector<std::size_t> order(count);
    std::mt19937_64 rng(seed);

    std::iota(order.begin(), order.end(), 0);
    shuffle(order, rng);
    return order;
}

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
    const MatrixSource trainSource(train);
    if (!validation)
        return fit(trainSource, nullptr, loss, optimizer, config, scheduler);
    const MatrixSource validationSource(*validation);
    return fit(trainSource, &validationSource, loss, optimizer, config, scheduler);
}

History Network::fit(const DataSource &train, const DataSource *validation, Loss &loss,
                     Optimizer &optimizer, const TrainConfig &config, const Scheduler *scheduler)
{
    checkSource(train, "training");
    if (validation) {
        checkSource(*validation, "validation");
        if (validation->inputWidth() != train.inputWidth() ||
            validation->targetWidth() != train.targetWidth())
            throw Error("fit: the validation set has other widths than the training set");
    }
    if (config.batchSize == 0)
        throw Error("fit: batch size must be positive");
    if (!(config.clipNorm >= 0) || !(config.clipValue >= 0))
        throw Error("fit: clip_norm and clip_value must be positive, or 0 for no clipping");
    if (_layers.empty())
        throw Error("fit: the network has no layer");
    if (config.patience > 0 && !validation)
        throw Error("fit: early stopping needs a validation set");
    const bool skipSoftmax = loss.fusesSoftmax();
    if (skipSoftmax && _layers.back()->type() != "softmax")
        throw Error("fit: " + loss.type() + " needs a final softmax layer");

    const std::size_t count = train.size();
    std::vector<std::size_t> order(count);
    std::mt19937_64 rng(config.seed);
    const std::vector<Parameter> params = parameters();
    std::vector<OwnedParameter> owned;
    for (std::size_t i = 0; i < _layers.size(); i++)
        for (const Parameter &p : _layers[i]->parameters())
            owned.push_back({p, i + 1, _layers[i]->type()});
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
            const std::vector<std::size_t> rows(order.begin() + static_cast<std::ptrdiff_t>(begin),
                                                order.begin() + static_cast<std::ptrdiff_t>(end));
            const auto [inputs, targets] = batchOf(train, rows);

            for (const Parameter &p : params)
                p.grad->fill(0);
            const Matrix prediction = forward(inputs, true);
            // The loss is a batch mean: weight it by the batch size so that a smaller last batch
            // does not count as much as a full one.
            const Scalar batchLoss = loss.value(prediction, targets);
            if (!std::isfinite(batchLoss))
                throw Error("fit: non-finite loss" + position(epoch, batches) +
                            "; lower the learning rate or set clip_norm / clip_value");
            lossSum += batchLoss * static_cast<Scalar>(end - begin);
            for (const auto &layer : _layers)
                penaltySum += layer->penalty();
            correct += countCorrect(prediction, targets);
            Matrix grad = loss.gradient(prediction, targets);
            for (std::size_t i = lastBackward; i-- > 0;)
                grad = _layers[i]->backward(grad);
            checkFinite(owned, true, epoch, batches);
            clip(owned, config);
            optimizer.step(params);
            checkFinite(owned, false, epoch, batches);
            batches++;
        }
        EpochStats stats;
        stats.trainLoss = lossSum / static_cast<Scalar>(count);
        stats.penalty = penaltySum / static_cast<Scalar>(batches);
        stats.trainAccuracy = static_cast<Scalar>(correct) / static_cast<Scalar>(count);
        if (validation) {
            // In chunks. Up to EVALUATION_CHUNK examples this is the loss of the whole set at
            // once; above, each chunk's mean loss is weighted by its size, as in training.
            const std::size_t total = validation->size();
            Scalar weightedSum = 0;
            std::size_t right = 0;
            for (std::size_t begin = 0; begin < total; begin += EVALUATION_CHUNK) {
                const std::size_t end = std::min(begin + EVALUATION_CHUNK, total);
                std::vector<std::size_t> rows(end - begin);
                std::iota(rows.begin(), rows.end(), begin);
                const auto [inputs, targets] = batchOf(*validation, rows);
                const Matrix predicted = predict(inputs);
                const Scalar chunkLoss = loss.value(predicted, targets);
                weightedSum += total <= EVALUATION_CHUNK
                                   ? chunkLoss
                                   : chunkLoss * static_cast<Scalar>(end - begin);
                right += countCorrect(predicted, targets);
            }
            stats.validationLoss = total <= EVALUATION_CHUNK
                                       ? weightedSum
                                       : weightedSum / static_cast<Scalar>(total);
            stats.validationAccuracy = static_cast<Scalar>(right) / static_cast<Scalar>(total);
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
