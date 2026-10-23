#include <criterion/criterion.h>

#include <cmath>
#include <memory>

#include "gradient_check.hpp"

using my_torch::BinaryCrossEntropy;
using my_torch::CrossEntropy;
using my_torch::Dense;
using my_torch::Huber;
using my_torch::MAE;
using my_torch::Matrix;
using my_torch::MSE;
using my_torch::ReLU;
using my_torch::Scalar;
using my_torch::Sigmoid;
using my_torch::Softmax;
using my_torch::Tanh;

namespace {

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows)
{
    Matrix m(rows.size(), rows.begin()->size());
    std::size_t r = 0;

    for (const auto &row : rows) {
        std::size_t c = 0;
        for (Scalar v : row)
            m(r, c++) = v;
        r++;
    }
    return m;
}

// Une distribution one-hot par ligne, la classe étant tirée parmi `classes`.
Matrix oneHot(std::size_t rows, std::size_t classes)
{
    Matrix m(rows, classes);

    for (std::size_t r = 0; r < rows; r++)
        m(r, (r * 7 + 1) % classes) = 1;
    return m;
}

gradient_check::Layers classifier(std::uint32_t seed, bool softmax)
{
    gradient_check::Layers layers;

    layers.push_back(std::make_unique<Dense>(5, 6, seed));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Dense>(6, 4, seed + 1));
    if (softmax)
        layers.push_back(std::make_unique<Softmax>());
    return layers;
}

} // namespace

Test(losses, mse_matches_hand_computation)
{
    const Matrix p = matrix({{1, 2}, {3, 0}});
    const Matrix t = matrix({{0, 2}, {1, 1}});
    const Matrix g = MSE().gradient(p, t);

    cr_assert_float_eq(MSE().value(p, t), (1 + 0 + 4 + 1) / 2.0, 1e-12);
    cr_assert_float_eq(g(0, 0), 1, 1e-12);
    cr_assert_float_eq(g(0, 1), 0, 1e-12);
    cr_assert_float_eq(g(1, 0), 2, 1e-12);
    cr_assert_float_eq(g(1, 1), -1, 1e-12);
}

Test(losses, mae_matches_hand_computation)
{
    const Matrix p = matrix({{1, 2}, {3, 0}});
    const Matrix t = matrix({{0, 2}, {1, 1}});
    const Matrix g = MAE().gradient(p, t);

    cr_assert_float_eq(MAE().value(p, t), (1 + 0 + 2 + 1) / 2.0, 1e-12);
    cr_assert_float_eq(g(0, 0), 0.5, 1e-12);
    cr_assert_float_eq(g(0, 1), 0, 1e-12);
    cr_assert_float_eq(g(1, 1), -0.5, 1e-12);
}

Test(losses, huber_is_quadratic_near_and_linear_far)
{
    const Matrix t = matrix({{0, 0}});
    const Huber huber(1.0);

    cr_assert_float_eq(huber.value(matrix({{0.5, 0}}), t), 0.125, 1e-12);
    cr_assert_float_eq(huber.value(matrix({{3, 0}}), t), 2.5, 1e-12);
    cr_assert_float_eq(huber.gradient(matrix({{3, 0.5}}), t)(0, 0), 1, 1e-12);
    cr_assert_float_eq(huber.gradient(matrix({{3, 0.5}}), t)(0, 1), 0.5, 1e-12);
    cr_assert_throw(Huber(0), my_torch::Error);
}

Test(losses, binary_cross_entropy_matches_hand_computation)
{
    const Matrix p = matrix({{0.8, 0.3}});
    const Matrix t = matrix({{1, 0}});

    cr_assert_float_eq(BinaryCrossEntropy().value(p, t), -std::log(0.8) - std::log(0.7), 1e-12);
}

Test(losses, cross_entropy_matches_hand_computation)
{
    const Matrix p = matrix({{0.7, 0.2, 0.1}, {0.25, 0.25, 0.5}});
    const Matrix t = matrix({{1, 0, 0}, {0, 0, 1}});
    const Matrix g = CrossEntropy().gradient(p, t);

    cr_assert_float_eq(CrossEntropy().value(p, t), (-std::log(0.7) - std::log(0.5)) / 2, 1e-12);
    cr_assert_float_eq(g(0, 0), -0.15, 1e-12);
    cr_assert_float_eq(g(0, 1), 0.1, 1e-12);
    cr_assert_float_eq(g(1, 2), -0.25, 1e-12);
}

Test(losses, cross_entropy_is_finite_when_the_true_class_has_zero_probability)
{
    const Matrix p = matrix({{0, 1}});
    const Matrix t = matrix({{1, 0}});
    const Matrix g = CrossEntropy().gradient(p, t);

    cr_assert(std::isfinite(CrossEntropy().value(p, t)));
    cr_assert_float_eq(g(0, 0), -1, 1e-12);
    cr_assert_float_eq(g(0, 1), 1, 1e-12);
}

Test(losses, label_smoothing_softens_the_target)
{
    const Matrix p = matrix({{0.5, 0.25, 0.25}});
    const Matrix t = matrix({{1, 0, 0}});
    const Matrix g = CrossEntropy(0.3).gradient(p, t);

    // q = [0.8, 0.1, 0.1] : le gradient vaut p - q.
    cr_assert_float_eq(g(0, 0), -0.3, 1e-12);
    cr_assert_float_eq(g(0, 1), 0.15, 1e-12);
    cr_assert_throw(CrossEntropy(1), my_torch::Error);
}

Test(losses, class_weights_scale_each_class)
{
    const Matrix p = matrix({{0.5, 0.5}, {0.5, 0.5}});
    const Matrix t = matrix({{1, 0}, {0, 1}});
    const CrossEntropy loss(0, {1, 3});

    cr_assert_float_eq(loss.value(p, t), (-std::log(0.5) - 3 * std::log(0.5)) / 2, 1e-12);
    // Exemple de la classe 1 (poids 3) : gradient 3·(p - t).
    cr_assert_float_eq(loss.gradient(p, t)(1, 1), 3 * (0.5 - 1) / 2, 1e-12);
    cr_assert_throw(CrossEntropy(0, {1, 2, 3}).value(p, t), my_torch::Error);
    cr_assert_throw(CrossEntropy(0, {1, -2}), my_torch::Error);
}

Test(losses, rejects_mismatched_shapes)
{
    cr_assert_throw(MSE().value(Matrix(2, 3), Matrix(2, 2)), my_torch::Error);
    cr_assert_throw(CrossEntropy().gradient(Matrix(1, 3), Matrix(2, 3)), my_torch::Error);
    cr_assert_throw(MAE().value(Matrix(0, 3), Matrix(0, 3)), my_torch::Error);
}

Test(losses, only_cross_entropy_fuses_softmax)
{
    cr_assert(CrossEntropy().fusesSoftmax());
    cr_assert(!MSE().fusesSoftmax());
    cr_assert(!BinaryCrossEntropy().fusesSoftmax());
}

Test(losses, standalone_gradients_match_finite_differences)
{
    const Matrix target = gradient_check::random(4, 3, 1);
    const Matrix prediction = gradient_check::random(4, 3, 2, 0.1, 0.9);
    const Matrix binary = oneHot(4, 3);

    cr_assert_lt(gradient_check::lossError(MSE(), prediction, target), gradient_check::tolerance);
    cr_assert_lt(gradient_check::lossError(Huber(0.3), prediction, target),
                 gradient_check::tolerance);
    cr_assert_lt(gradient_check::lossError(BinaryCrossEntropy(), prediction, binary),
                 gradient_check::tolerance);
}

Test(gradient_check, mse_through_a_full_network)
{
    gradient_check::Layers layers;
    layers.push_back(std::make_unique<Dense>(5, 6, 1));
    layers.push_back(std::make_unique<Tanh>());
    layers.push_back(std::make_unique<Dense>(6, 2, 2));

    cr_assert_lt(gradient_check::networkError(layers, MSE(), gradient_check::random(7, 5, 3),
                                              gradient_check::random(7, 2, 4)),
                 gradient_check::tolerance);
}

Test(gradient_check, binary_cross_entropy_through_a_full_network)
{
    gradient_check::Layers layers;
    layers.push_back(std::make_unique<Dense>(5, 6, 1));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Dense>(6, 3, 2));
    layers.push_back(std::make_unique<Sigmoid>());

    cr_assert_lt(gradient_check::networkError(layers, BinaryCrossEntropy(),
                                              gradient_check::random(7, 5, 3), oneHot(7, 3)),
                 gradient_check::tolerance);
}

Test(gradient_check, cross_entropy_through_a_full_network)
{
    auto layers = classifier(1, true);

    cr_assert_lt(gradient_check::networkError(layers, CrossEntropy(),
                                              gradient_check::random(8, 5, 3), oneHot(8, 4)),
                 gradient_check::tolerance);
}

Test(gradient_check, cross_entropy_with_smoothing_and_weights_through_a_full_network)
{
    auto layers = classifier(5, true);
    const CrossEntropy loss(0.1, {1, 2, 0.5, 4});

    cr_assert_lt(
        gradient_check::networkError(layers, loss, gradient_check::random(8, 5, 3), oneHot(8, 4)),
        gradient_check::tolerance);
}

// Le contrôle doit détecter un gradient faux : sans le saut du Softmax, le gradient fusionné
// (p - q) passerait dans le backward du Softmax et serait faux.
Test(gradient_check, detects_a_wrong_gradient)
{
    struct Unfused : CrossEntropy {
        bool fusesSoftmax() const noexcept override
        {
            return false;
        }
    };
    auto layers = classifier(1, true);

    cr_assert_gt(gradient_check::networkError(layers, Unfused(), gradient_check::random(8, 5, 3),
                                              oneHot(8, 4)),
                 1e-3);
}
