// Container layers: residual blocks and branches side by side (ADR-0046).
#include "my_torch/Containers.hpp"

#include <algorithm>
#include <string>

#include "my_torch/Error.hpp"
#include "my_torch/Normalization.hpp"

namespace my_torch {

std::size_t sequenceWidth(const LayerSequence &layers, std::size_t inputWidth)
{
    std::size_t width = inputWidth;
    for (std::size_t i = 0; i < layers.size(); i++) {
        std::size_t normalized = 0;
        if (const auto *batch = dynamic_cast<const BatchNorm *>(layers[i].get()))
            normalized = batch->features();
        if (const auto *norm = dynamic_cast<const LayerNorm *>(layers[i].get()))
            normalized = norm->features();
        if (normalized != 0 && normalized != width)
            throw Error("layer " + std::to_string(i + 1) + " normalizes " +
                        std::to_string(normalized) + " features but receives " +
                        std::to_string(width));
        try {
            width = layers[i]->outputWidth(width);
        } catch (const Error &error) {
            throw Error("layer " + std::to_string(i + 1) + " " + error.what());
        }
    }
    return width;
}

namespace {

Matrix forwardAll(LayerSequence &layers, const Matrix &input, bool training)
{
    Matrix x = input;
    for (auto &layer : layers)
        x = layer->forward(x, training);
    return x;
}

Matrix forwardAllMonteCarlo(LayerSequence &layers, const Matrix &input)
{
    Matrix x = input;
    for (auto &layer : layers)
        x = layer->forwardMonteCarlo(x);
    return x;
}

Matrix backwardAll(LayerSequence &layers, const Matrix &gradOutput)
{
    Matrix g = gradOutput;
    for (std::size_t i = layers.size(); i-- > 0;)
        g = layers[i]->backward(g);
    return g;
}

void appendParameters(LayerSequence &layers, std::vector<Parameter> &out)
{
    for (auto &layer : layers)
        for (const Parameter &p : layer->parameters())
            out.push_back(p);
}

Scalar penaltyOf(const LayerSequence &layers)
{
    Scalar sum = 0;
    for (const auto &layer : layers)
        sum += layer->penalty();
    return sum;
}

// Each layer, then the layers it holds, depth first.
void appendChildren(const LayerSequence &layers, std::vector<Layer *> &out)
{
    for (const auto &layer : layers) {
        out.push_back(layer.get());
        for (Layer *child : layer->children())
            out.push_back(child);
    }
}

Matrix sum(Matrix a, const Matrix &b)
{
    a += b;
    return a;
}

} // namespace

// ---- Residual ----------------------------------------------------------------------------------

Residual::Residual(LayerSequence layers) : _layers(std::move(layers))
{
    if (_layers.empty())
        throw Error("residual: the block needs at least one layer");
}

Matrix Residual::forward(const Matrix &input, bool training)
{
    const Matrix out = forwardAll(_layers, input, training);
    if (out.rows() != input.rows() || out.cols() != input.cols())
        throw Error("residual: the block gives " + std::to_string(out.cols()) + " columns for " +
                    std::to_string(input.cols()));
    return sum(out, input);
}

Matrix Residual::forwardMonteCarlo(const Matrix &input)
{
    return sum(forwardAllMonteCarlo(_layers, input), input);
}

Matrix Residual::backward(const Matrix &gradOutput)
{
    return sum(backwardAll(_layers, gradOutput), gradOutput);
}

std::vector<Parameter> Residual::parameters()
{
    std::vector<Parameter> out;
    appendParameters(_layers, out);
    return out;
}

Scalar Residual::penalty() const
{
    return penaltyOf(_layers);
}

void Residual::save(std::ostream &out) const
{
    out << "residual layers=" << _layers.size() << '\n';
    for (const auto &layer : _layers)
        layer->save(out);
}

void Residual::saveState(std::ostream &out) const
{
    for (const auto &layer : _layers)
        layer->saveState(out);
}

void Residual::loadState(std::istream &in)
{
    for (auto &layer : _layers)
        layer->loadState(in);
}

std::size_t Residual::outputWidth(std::size_t inputWidth) const
{
    std::size_t width = 0;
    try {
        width = sequenceWidth(_layers, inputWidth);
    } catch (const Error &error) {
        throw Error(std::string("residual: ") + error.what());
    }
    if (width != inputWidth)
        throw Error("residual: the block gives " + std::to_string(width) + " columns for " +
                    std::to_string(inputWidth) + " (the shortcut needs the same width)");
    return inputWidth;
}

std::string Residual::type() const
{
    return "residual";
}

std::vector<Layer *> Residual::children() const
{
    std::vector<Layer *> out;
    appendChildren(_layers, out);
    return out;
}

// ---- Concat ------------------------------------------------------------------------------------

Concat::Concat(std::vector<LayerSequence> branches) : _branches(std::move(branches))
{
    if (_branches.size() < 2)
        throw Error("concat: at least two branches are needed");
    for (const LayerSequence &branch : _branches)
        if (branch.empty())
            throw Error("concat: every branch needs at least one layer");
}

Matrix Concat::join(const std::vector<Matrix> &outputs)
{
    _widths.clear();
    std::size_t total = 0;
    for (const Matrix &out : outputs) {
        if (out.rows() != outputs.front().rows())
            throw Error("concat: the branches give different row counts");
        _widths.push_back(out.cols());
        total += out.cols();
    }
    Matrix joined(outputs.front().rows(), total);
    for (std::size_t r = 0; r < joined.rows(); r++) {
        std::size_t column = 0;
        for (const Matrix &out : outputs) {
            std::copy_n(out.data() + r * out.cols(), out.cols(), &joined(r, column));
            column += out.cols();
        }
    }
    return joined;
}

Matrix Concat::forward(const Matrix &input, bool training)
{
    std::vector<Matrix> outputs;
    for (LayerSequence &branch : _branches)
        outputs.push_back(forwardAll(branch, input, training));
    return join(outputs);
}

Matrix Concat::forwardMonteCarlo(const Matrix &input)
{
    std::vector<Matrix> outputs;
    for (LayerSequence &branch : _branches)
        outputs.push_back(forwardAllMonteCarlo(branch, input));
    return join(outputs);
}

Matrix Concat::backward(const Matrix &gradOutput)
{
    if (_widths.size() != _branches.size())
        throw Error("concat: backward before forward");
    std::size_t total = 0;
    for (const std::size_t w : _widths)
        total += w;
    if (gradOutput.cols() != total)
        throw Error("concat: the gradient has " + std::to_string(gradOutput.cols()) +
                    " columns for " + std::to_string(total) + " outputs");
    Matrix gradInput;
    std::size_t column = 0;
    for (std::size_t b = 0; b < _branches.size(); b++) {
        Matrix slice(gradOutput.rows(), _widths[b]);
        for (std::size_t r = 0; r < slice.rows(); r++)
            std::copy_n(gradOutput.data() + r * gradOutput.cols() + column, _widths[b],
                        &slice(r, 0));
        column += _widths[b];
        const Matrix g = backwardAll(_branches[b], slice);
        if (b == 0)
            gradInput = g;
        else
            gradInput += g;
    }
    return gradInput;
}

std::vector<Parameter> Concat::parameters()
{
    std::vector<Parameter> out;
    for (LayerSequence &branch : _branches)
        appendParameters(branch, out);
    return out;
}

Scalar Concat::penalty() const
{
    Scalar total = 0;
    for (const LayerSequence &branch : _branches)
        total += penaltyOf(branch);
    return total;
}

void Concat::save(std::ostream &out) const
{
    out << "concat branches=" << _branches.size() << '\n';
    for (const LayerSequence &branch : _branches) {
        out << "branch layers=" << branch.size() << '\n';
        for (const auto &layer : branch)
            layer->save(out);
    }
}

void Concat::saveState(std::ostream &out) const
{
    for (const LayerSequence &branch : _branches)
        for (const auto &layer : branch)
            layer->saveState(out);
}

void Concat::loadState(std::istream &in)
{
    for (LayerSequence &branch : _branches)
        for (auto &layer : branch)
            layer->loadState(in);
}

std::size_t Concat::outputWidth(std::size_t inputWidth) const
{
    std::size_t total = 0;
    for (std::size_t b = 0; b < _branches.size(); b++) {
        try {
            total += sequenceWidth(_branches[b], inputWidth);
        } catch (const Error &error) {
            throw Error("concat: branch " + std::to_string(b + 1) + ": " + error.what());
        }
    }
    return total;
}

std::string Concat::type() const
{
    return "concat";
}

std::vector<Layer *> Concat::children() const
{
    std::vector<Layer *> out;
    for (const LayerSequence &branch : _branches)
        appendChildren(branch, out);
    return out;
}

} // namespace my_torch
