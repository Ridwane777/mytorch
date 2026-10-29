#include "my_torch/Network.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>

#include "my_torch/Checkpoint.hpp"
#include "my_torch/Dense.hpp"
#include "my_torch/Dropout.hpp"
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

// What a checkpoint must match (ADR-0032): every training setting but the number of epochs, so
// that a finished training can be extended, the layers and their shapes, the optimizer, and the
// sizes of the data. The data itself is not hashed: reading it all would cost a pass over the file.
std::uint64_t fingerprint(const std::vector<std::unique_ptr<Layer>> &layers,
                          const Optimizer &optimizer, const TrainConfig &c,
                          const DataSource &train, const DataSource *validation)
{
    std::ostringstream text;
    text.precision(17);
    text << "batch=" << c.batchSize << " seed=" << c.seed << " shuffle=" << c.shuffle
         << " patience=" << c.patience << " min_delta=" << c.minDelta
         << " restore_best=" << c.restoreBest << " clip_norm=" << c.clipNorm
         << " clip_value=" << c.clipValue << " optimizer=" << optimizer.type()
         << " train=" << train.size() << 'x' << train.inputWidth() << 'x' << train.targetWidth()
         << " validation=" << (validation ? validation->size() : 0) << '\n';
    for (const auto &layer : layers) {
        text << layer->type();
        for (const Parameter &p : layer->parameters())
            text << ' ' << p.value->rows() << 'x' << p.value->cols();
        text << '\n';
    }
    return fingerprintOf(text.str());
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

Matrix Network::inputGradient(const Matrix &input, std::size_t output)
{
    const std::vector<Parameter> params = parameters();
    std::vector<Matrix> savedGradients;
    for (const Parameter &p : params)
        savedGradients.push_back(*p.grad);
    const Matrix probabilities = forward(input, false);
    if (output >= probabilities.cols())
        throw Error("inputGradient: output " + std::to_string(output) + " out of range");
    Matrix grad(probabilities.rows(), probabilities.cols());
    for (std::size_t r = 0; r < grad.rows(); r++)
        grad(r, output) = 1;
    for (std::size_t i = _layers.size(); i-- > 0;)
        grad = _layers[i]->backward(grad);
    for (std::size_t i = 0; i < params.size(); i++)
        *params[i].grad = savedGradients[i];
    return grad;
}

std::string modelSummary(Network &network)
{
    std::ostringstream out;
    std::size_t width = network.info().inputs;
    std::size_t total = 0;
    out << "#   layer          output  parameters\n";
    for (std::size_t i = 0; i < network.layers().size(); i++) {
        Layer &layer = *network.layers()[i];
        std::size_t count = 0;
        for (const Parameter &p : layer.parameters())
            count += p.value->size();
        if (const auto *dense = dynamic_cast<const Dense *>(&layer))
            width = dense->weights().cols();
        total += count;
        char line[96];
        std::snprintf(line, sizeof line, "%-3zu %-14s %6zu  %10zu\n", i + 1, layer.type().c_str(),
                      width, count);
        out << line;
    }
    out << "total parameters: " << total << '\n';
    return out.str();
}

UncertainPrediction Network::predictMonteCarlo(const Matrix &input, std::size_t passes,
                                               std::uint64_t seed)
{
    if (passes == 0)
        throw Error("predictMonteCarlo: at least one pass is needed");
    std::vector<std::size_t> dropouts;
    for (std::size_t i = 0; i < _layers.size(); i++)
        if (dynamic_cast<const Dropout *>(_layers[i].get()))
            dropouts.push_back(i);
    if (dropouts.empty())
        throw Error("predictMonteCarlo: the network has no dropout layer, every pass would be the "
                    "same");
    // Keep the generators of the network, and give each dropout layer one derived from seed.
    std::vector<std::string> saved;
    for (const std::size_t i : dropouts) {
        std::ostringstream state;
        _layers[i]->saveState(state);
        saved.push_back(state.str());
        std::ostringstream fresh;
        fresh << std::mt19937_64(seed * 1000003 + i);
        std::istringstream in(fresh.str());
        _layers[i]->loadState(in);
    }

    const std::size_t rows = input.rows();
    UncertainPrediction result;
    std::vector<Scalar> passEntropy(rows, 0);
    std::vector<std::vector<std::size_t>> votes(rows);
    for (std::size_t pass = 0; pass < passes; pass++) {
        Matrix output = input;
        for (const auto &layer : _layers)
            output = layer->forward(output, dynamic_cast<const Dropout *>(layer.get()) != nullptr);
        if (pass == 0) {
            result.mean = Matrix(rows, output.cols());
            for (auto &v : votes)
                v.assign(output.cols(), 0);
        }
        for (std::size_t r = 0; r < rows; r++) {
            std::size_t top = 0;
            for (std::size_t c = 0; c < output.cols(); c++) {
                const Scalar p = output(r, c);
                result.mean(r, c) += p / static_cast<Scalar>(passes);
                if (p > 0)
                    passEntropy[r] -= p * std::log(p) / static_cast<Scalar>(passes);
                if (p > output(r, top))
                    top = c;
            }
            votes[r][top]++;
        }
    }
    for (std::size_t k = 0; k < dropouts.size(); k++) {
        std::istringstream in(saved[k]);
        _layers[dropouts[k]]->loadState(in);
    }

    for (std::size_t r = 0; r < rows; r++) {
        Scalar entropy = 0;
        for (std::size_t c = 0; c < result.mean.cols(); c++) {
            const Scalar p = result.mean(r, c);
            if (p > 0)
                entropy -= p * std::log(p);
        }
        const std::size_t mode = *std::max_element(votes[r].begin(), votes[r].end());
        result.entropy.push_back(entropy);
        result.mutualInformation.push_back(std::max<Scalar>(0, entropy - passEntropy[r]));
        result.disagreement.push_back(1 - static_cast<Scalar>(mode) / static_cast<Scalar>(passes));
    }
    return result;
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
                     Optimizer &optimizer, const TrainConfig &config, const Scheduler *scheduler,
                     const std::string &checkpoint)
{
    const MatrixSource trainSource(train);
    if (!validation)
        return fit(trainSource, nullptr, loss, optimizer, config, scheduler, checkpoint);
    const MatrixSource validationSource(*validation);
    return fit(trainSource, &validationSource, loss, optimizer, config, scheduler, checkpoint);
}

History Network::fit(const DataSource &train, const DataSource *validation, Loss &loss,
                     Optimizer &optimizer, const TrainConfig &config, const Scheduler *scheduler,
                     const std::string &checkpoint)
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
    // The layers' own state at the best epoch: restoring the weights of an epoch with the running
    // averages of a later one would break a BatchNorm network (ADR-0033).
    std::vector<std::string> bestLayers;
    const auto layerStates = [&] {
        std::vector<std::string> states;
        for (const auto &layer : _layers) {
            std::ostringstream text;
            text.precision(17);
            layer->saveState(text);
            states.push_back(text.str());
        }
        return states;
    };
    std::size_t start = 0;
    const std::uint64_t print =
        checkpoint.empty() ? 0 : fingerprint(_layers, optimizer, config, train, validation);

    if (!checkpoint.empty())
        if (const std::optional<TrainingState> state = loadCheckpoint(checkpoint, print)) {
            if (state->weights.size() != params.size() || state->layers.size() != _layers.size())
                throw Error("checkpoint '" + checkpoint + "' does not match the network");
            for (std::size_t i = 0; i < params.size(); i++) {
                if (state->weights[i].rows() != params[i].value->rows() ||
                    state->weights[i].cols() != params[i].value->cols())
                    throw Error("checkpoint '" + checkpoint + "' does not match the network");
                *params[i].value = state->weights[i];
            }
            std::istringstream optimizerState(state->optimizer);
            optimizer.loadState(optimizerState);
            for (std::size_t i = 0; i < _layers.size(); i++) {
                std::istringstream layerState(state->layers[i]);
                _layers[i]->loadState(layerState);
            }
            std::istringstream shuffleState(state->shuffle);
            if (!(shuffleState >> rng))
                throw Error("checkpoint '" + checkpoint + "': malformed generator state");
            history = state->history;
            best = state->best;
            sinceBest = state->sinceBest;
            bestWeights = state->bestWeights;
            bestLayers = state->bestLayers;
            start = state->epoch;
        }
    // Saves the state after `finished` epochs.
    const auto save = [&](std::size_t finished) {
        TrainingState state;
        state.epoch = finished;
        std::ostringstream shuffleState;
        shuffleState << rng;
        state.shuffle = shuffleState.str();
        state.history = history;
        state.best = best;
        state.sinceBest = sinceBest;
        state.bestWeights = bestWeights;
        state.bestLayers = bestLayers;
        for (const Parameter &p : params)
            state.weights.push_back(*p.value);
        std::ostringstream optimizerState;
        optimizerState.precision(17);
        optimizer.saveState(optimizerState);
        state.optimizer = optimizerState.str();
        for (const auto &layer : _layers) {
            std::ostringstream layerState;
            layerState.precision(17);
            layer->saveState(layerState);
            state.layers.push_back(layerState.str());
        }
        saveCheckpoint(checkpoint, print, state);
    };
    // A checkpoint written when early stopping ended the training: nothing left to run.
    const bool stopped = config.patience > 0 && sinceBest >= config.patience;

    for (std::size_t epoch = start; epoch < config.epochs && !stopped; epoch++) {
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
        if (config.onEpoch)
            config.onEpoch(epoch, stats);
        bool stop = false;
        if (config.patience > 0) {
            if (bestWeights.empty() || stats.validationLoss < best - config.minDelta) {
                best = stats.validationLoss;
                sinceBest = 0;
                bestWeights.clear();
                for (const Parameter &p : params)
                    bestWeights.push_back(*p.value);
                bestLayers = layerStates();
            } else {
                stop = ++sinceBest >= config.patience;
            }
        }
        if (!checkpoint.empty())
            save(epoch + 1);
        if (stop)
            break;
    }
    if (config.restoreBest && !bestWeights.empty()) {
        for (std::size_t i = 0; i < params.size(); i++)
            *params[i].value = bestWeights[i];
        for (std::size_t i = 0; i < bestLayers.size() && i < _layers.size(); i++) {
            std::istringstream text(bestLayers[i]);
            _layers[i]->loadState(text);
        }
    }
    return history;
}

} // namespace my_torch
