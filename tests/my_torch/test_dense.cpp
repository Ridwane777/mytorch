#include <criterion/criterion.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "gradient_check.hpp"

using my_torch::Dense;
using my_torch::Init;
using my_torch::Matrix;

namespace {

// The 2 × 2 example from the course: X = [[1, 2], [3, 0]], W = [[0.5, -1], [0.5, 1]], b = [0.1,
// 0.2].
Dense example()
{
    Dense dense(2, 2);

    dense.weights()(0, 0) = 0.5;
    dense.weights()(0, 1) = -1;
    dense.weights()(1, 0) = 0.5;
    dense.weights()(1, 1) = 1;
    dense.bias()(0, 0) = 0.1;
    dense.bias()(0, 1) = 0.2;
    return dense;
}

Matrix exampleInput()
{
    Matrix x(2, 2);

    x(0, 0) = 1;
    x(0, 1) = 2;
    x(1, 0) = 3;
    return x;
}

} // namespace

Test(dense, forward_matches_hand_computation)
{
    Dense dense = example();
    Matrix z = dense.forward(exampleInput(), true);

    cr_assert_float_eq(z(0, 0), 1.6, 1e-12);
    cr_assert_float_eq(z(0, 1), 1.2, 1e-12);
    cr_assert_float_eq(z(1, 0), 1.6, 1e-12);
    cr_assert_float_eq(z(1, 1), -2.8, 1e-12);
}

Test(dense, backward_matches_hand_computation)
{
    Dense dense = example();
    Matrix g(2, 2);
    g(0, 0) = 1;
    g(1, 1) = 1;

    dense.forward(exampleInput(), true);
    Matrix dx = dense.backward(g);
    auto params = dense.parameters();
    const Matrix &dw = *params[0].grad;
    const Matrix &db = *params[1].grad;

    cr_assert_eq(dw(0, 0), 1);
    cr_assert_eq(dw(0, 1), 3);
    cr_assert_eq(dw(1, 0), 2);
    cr_assert_eq(dw(1, 1), 0);
    cr_assert_eq(db(0, 0), 1);
    cr_assert_eq(db(0, 1), 1);
    cr_assert_eq(dx(0, 0), 0.5);
    cr_assert_eq(dx(0, 1), 0.5);
    cr_assert_eq(dx(1, 0), -1);
    cr_assert_eq(dx(1, 1), 1);
}

Test(dense, gradients_accumulate_across_backward_calls)
{
    Dense dense = example();
    Matrix g(2, 2, 1);

    dense.forward(exampleInput(), true);
    dense.backward(g);
    dense.backward(g);
    cr_assert_eq((*dense.parameters()[1].grad)(0, 0), 4);
}

Test(dense, initialisation_is_bounded_and_reproducible)
{
    Dense a(768, 128);
    Dense b(768, 128);
    const double limit = std::sqrt(1.0 / 768);

    for (std::size_t i = 0; i < a.weights().size(); i++) {
        cr_assert_leq(std::abs(a.weights().data()[i]), limit);
        cr_assert_eq(a.weights().data()[i], b.weights().data()[i]);
    }
    cr_assert_eq(a.bias()(0, 0), 0);
}

Test(dense, rejects_wrong_input_width)
{
    Dense dense(3, 2);

    cr_assert_throw(dense.forward(Matrix(1, 4), true), my_torch::Error);
}

Test(dense, gradients_match_finite_differences)
{
    Dense dense(4, 3, 1);
    Matrix input = gradient_check::random(5, 4, 2);
    const Matrix weights = gradient_check::random(5, 3, 3);

    dense.forward(input, true);
    dense.backward(weights);
    auto params = dense.parameters();
    for (const auto &p : params)
        cr_assert_lt(gradient_check::maxError(dense, input, weights, *p.value, *p.grad),
                     gradient_check::tolerance);
    cr_assert_lt(gradient_check::inputError(dense, input, 3), gradient_check::tolerance);
}

namespace {

struct Moments {
    double mean;
    double variance;
    double maxAbs;
};

Moments moments(const Matrix &w)
{
    double sum = 0;
    double square = 0;
    double maxAbs = 0;

    for (std::size_t i = 0; i < w.size(); i++) {
        sum += w.data()[i];
        square += w.data()[i] * w.data()[i];
        maxAbs = std::max(maxAbs, std::abs(w.data()[i]));
    }
    const double n = static_cast<double>(w.size());
    return {sum / n, square / n - (sum / n) * (sum / n), maxAbs};
}

// Largest |WᵀW - I| (columns = true) or |WWᵀ - I| (columns = false).
double orthogonalityError(const Matrix &w, bool columns)
{
    const Matrix product = columns ? w.transpose().matmul(w) : w.matmul(w.transpose());
    double error = 0;

    for (std::size_t r = 0; r < product.rows(); r++)
        for (std::size_t c = 0; c < product.cols(); c++)
            error = std::max(error, std::abs(product(r, c) - (r == c ? 1.0 : 0.0)));
    return error;
}

struct Scheme {
    Init init;
    bool uniform;
    double variance; // target, for a 400 × 300 layer
};

constexpr std::size_t bigIn = 400;
constexpr std::size_t bigOut = 300;

} // namespace

Test(dense_init, variance_and_mean_match_the_target)
{
    const double lecun = 1.0 / bigIn;
    const double xavier = 2.0 / (bigIn + bigOut);
    const double he = 2.0 / bigIn;
    const std::vector<Scheme> schemes = {
        {Init::LecunUniform, true, lecun},   {Init::LecunNormal, false, lecun},
        {Init::XavierUniform, true, xavier}, {Init::XavierNormal, false, xavier},
        {Init::HeUniform, true, he},         {Init::HeNormal, false, he},
    };
    for (const Scheme &scheme : schemes) {
        Dense dense(bigIn, bigOut, 7, {}, scheme.init);
        const Moments m = moments(dense.weights());

        cr_assert_lt(std::abs(m.mean), 0.02 * std::sqrt(scheme.variance), "mean %g", m.mean);
        cr_assert_lt(std::abs(m.variance / scheme.variance - 1), 0.05, "variance %g vs %g",
                     m.variance, scheme.variance);
        if (scheme.uniform)
            cr_assert_leq(m.maxAbs, std::sqrt(3 * scheme.variance));
    }
}

Test(dense_init, uniform_keeps_the_legacy_bound_and_variance)
{
    Dense dense(bigIn, bigOut, 7, {}, Init::Uniform);
    const Moments m = moments(dense.weights());
    const double limit = std::sqrt(1.0 / bigIn);

    cr_assert_leq(m.maxAbs, limit);
    cr_assert_lt(std::abs(m.variance * 3 * bigIn - 1), 0.05, "variance %g", m.variance);
    cr_assert_lt(std::abs(m.mean), 0.02 * limit);
}

Test(dense_init, default_is_uniform_with_the_same_draws)
{
    const Dense implicit(40, 30, 5);
    const Dense named(40, 30, 5, {}, Init::Uniform);

    for (std::size_t i = 0; i < implicit.weights().size(); i++)
        cr_assert_eq(implicit.weights().data()[i], named.weights().data()[i]);
}

Test(dense_init, orthogonal_has_orthonormal_columns_when_tall)
{
    Dense dense(64, 32, 3, {}, Init::Orthogonal);

    cr_assert_lt(orthogonalityError(dense.weights(), true), 1e-9);
}

Test(dense_init, orthogonal_has_orthonormal_rows_when_wide)
{
    Dense dense(32, 64, 3, {}, Init::Orthogonal);

    cr_assert_lt(orthogonalityError(dense.weights(), false), 1e-9);
}

Test(dense_init, orthogonal_is_an_orthogonal_matrix_when_square)
{
    Dense dense(48, 48, 3, {}, Init::Orthogonal);

    cr_assert_lt(orthogonalityError(dense.weights(), true), 1e-9);
    cr_assert_lt(orthogonalityError(dense.weights(), false), 1e-9);
}

Test(dense_init, orthogonal_handles_a_single_row_and_column)
{
    Dense row(1, 5, 3, {}, Init::Orthogonal);
    Dense column(5, 1, 3, {}, Init::Orthogonal);

    cr_assert_lt(orthogonalityError(row.weights(), false), 1e-9);
    cr_assert_lt(orthogonalityError(column.weights(), true), 1e-9);
}

Test(dense_init, same_seed_same_weights_other_seed_other_weights)
{
    for (Init init : {Init::Uniform, Init::LecunUniform, Init::LecunNormal, Init::XavierUniform,
                      Init::XavierNormal, Init::HeUniform, Init::HeNormal, Init::Orthogonal}) {
        const Dense a(20, 10, 11, {}, init);
        const Dense b(20, 10, 11, {}, init);
        const Dense c(20, 10, 12, {}, init);
        bool differs = false;

        for (std::size_t i = 0; i < a.weights().size(); i++) {
            cr_assert_eq(a.weights().data()[i], b.weights().data()[i]);
            differs = differs || a.weights().data()[i] != c.weights().data()[i];
        }
        cr_assert(differs);
    }
}

Test(dense_init, biases_start_at_zero)
{
    for (Init init : {Init::Uniform, Init::HeNormal, Init::XavierUniform, Init::Orthogonal}) {
        Dense dense(8, 6, 1, {}, init);

        for (std::size_t c = 0; c < 6; c++)
            cr_assert_eq(dense.bias()(0, c), 0);
    }
}
