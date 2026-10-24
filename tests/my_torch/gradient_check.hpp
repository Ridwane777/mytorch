#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "my_torch/my_torch.hpp"

// Vérification des gradients par différences finies centrées. On prend comme coût
// L = Σ sortie ⊙ R, avec R fixé au hasard : dL/dsortie vaut alors exactement R, que l'on donne
// à backward. Chaque nombre de la matrice testée est ensuite bougé de ±h et la pente mesurée
// (L(+h) - L(-h)) / 2h est comparée au gradient calculé par backward.
namespace gradient_check {

using my_torch::Layer;
using my_torch::Matrix;
using my_torch::Scalar;

inline constexpr Scalar step = 1e-6;
inline constexpr Scalar tolerance = 1e-6;

inline Matrix random(std::size_t rows, std::size_t cols, std::uint32_t seed, Scalar low = -1,
                     Scalar high = 1)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Scalar> dist(low, high);
    Matrix m(rows, cols);

    for (std::size_t i = 0; i < m.size(); i++)
        m.data()[i] = dist(rng);
    return m;
}

inline Scalar loss(Layer &layer, const Matrix &input, const Matrix &weights)
{
    const Matrix out = layer.forward(input, true);
    Scalar sum = 0;

    for (std::size_t i = 0; i < out.size(); i++)
        sum += out.data()[i] * weights.data()[i];
    return sum;
}

// Plus grand écart relatif entre analytic et la pente mesurée en bougeant chaque case de target.
inline Scalar maxError(Layer &layer, const Matrix &input, const Matrix &weights, Matrix &target,
                       const Matrix &analytic)
{
    Scalar worst = 0;

    for (std::size_t i = 0; i < target.size(); i++) {
        const Scalar saved = target.data()[i];
        target.data()[i] = saved + step;
        const Scalar plus = loss(layer, input, weights);
        target.data()[i] = saved - step;
        const Scalar minus = loss(layer, input, weights);
        target.data()[i] = saved;
        const Scalar numeric = (plus - minus) / (2 * step);
        const Scalar a = analytic.data()[i];
        worst =
            std::max(worst, std::abs(a - numeric) / std::max(1.0, std::abs(a) + std::abs(numeric)));
    }
    return worst;
}

// Gradient par rapport à l'entrée : vérifie le dX renvoyé par backward.
inline Scalar inputError(Layer &layer, Matrix input, std::size_t outputs, std::uint32_t seed = 7)
{
    const Matrix weights = random(input.rows(), outputs, seed);
    layer.forward(input, true);
    const Matrix analytic = layer.backward(weights);
    return maxError(layer, input, weights, input, analytic);
}

// Gradient of a loss with respect to the prediction (for losses not coupled to Softmax).
inline Scalar lossError(const my_torch::Loss &loss, Matrix prediction, const Matrix &target)
{
    const Matrix analytic = loss.gradient(prediction, target);
    Scalar worst = 0;

    for (std::size_t i = 0; i < prediction.size(); i++) {
        const Scalar saved = prediction.data()[i];
        prediction.data()[i] = saved + step;
        const Scalar plus = loss.value(prediction, target);
        prediction.data()[i] = saved - step;
        const Scalar minus = loss.value(prediction, target);
        prediction.data()[i] = saved;
        const Scalar numeric = (plus - minus) / (2 * step);
        const Scalar a = analytic.data()[i];
        worst = std::max(worst,
                         std::abs(a - numeric) / std::max(1e-10, std::abs(a) + std::abs(numeric)));
    }
    return worst;
}

using Layers = std::vector<std::unique_ptr<Layer>>;

inline Matrix forward(Layers &layers, const Matrix &input)
{
    Matrix x = input;

    for (auto &layer : layers)
        x = layer->forward(x, true);
    return x;
}

// Checks the gradients of every parameter of a complete network, from the prediction to the loss:
// backpropagation written by hand like Network::fit (the final Softmax backward is skipped when
// the loss fuses it), then compared with the finite differences of loss.value. Returns the largest
// relative error over all weights and biases.
inline Scalar networkError(Layers &layers, const my_torch::Loss &loss, const Matrix &input,
                           const Matrix &target)
{
    if (loss.fusesSoftmax() && (layers.empty() || layers.back()->type() != "softmax"))
        throw std::logic_error("a loss that fuses Softmax needs a final Softmax layer");
    std::vector<my_torch::Parameter> params;

    for (auto &layer : layers)
        for (const auto &p : layer->parameters()) {
            p.grad->fill(0);
            params.push_back(p);
        }
    Matrix grad = loss.gradient(forward(layers, input), target);
    for (std::size_t i = layers.size(); i-- > 0;) {
        if (i + 1 == layers.size() && loss.fusesSoftmax() && layers[i]->type() == "softmax")
            continue;
        grad = layers[i]->backward(grad);
    }
    Scalar worst = 0;
    for (const auto &p : params) {
        for (std::size_t i = 0; i < p.value->size(); i++) {
            const Scalar saved = p.value->data()[i];
            p.value->data()[i] = saved + step;
            const Scalar plus = loss.value(forward(layers, input), target);
            p.value->data()[i] = saved - step;
            const Scalar minus = loss.value(forward(layers, input), target);
            p.value->data()[i] = saved;
            const Scalar numeric = (plus - minus) / (2 * step);
            const Scalar a = p.grad->data()[i];
            worst = std::max(worst, std::abs(a - numeric) /
                                        std::max(1e-10, std::abs(a) + std::abs(numeric)));
        }
    }
    return worst;
}

} // namespace gradient_check
