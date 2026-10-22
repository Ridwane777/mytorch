#pragma once

#include <cmath>
#include <functional>
#include <random>

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
        worst = std::max(worst, std::abs(a - numeric) / std::max(1.0, std::abs(a) + std::abs(numeric)));
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

} // namespace gradient_check
