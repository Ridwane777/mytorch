#include <criterion/criterion.h>

#include "gradient_check.hpp"

using my_torch::Dense;
using my_torch::Matrix;

namespace {

// L'exemple 2 × 2 du cours : X = [[1, 2], [3, 0]], W = [[0.5, -1], [0.5, 1]], b = [0.1, 0.2].
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
