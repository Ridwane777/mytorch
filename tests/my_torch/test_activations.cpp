#include <criterion/criterion.h>

#include <cmath>

#include "gradient_check.hpp"

using namespace my_torch;
using gradient_check::inputError;
using gradient_check::tolerance;

namespace {

// Values kept away from 0: ReLU and Leaky ReLU are not differentiable there.
Matrix awayFromZero()
{
    return gradient_check::random(4, 5, 11).map([](Scalar z) { return z + (z >= 0 ? 0.1 : -0.1); });
}

} // namespace

Test(activations, relu_forward_and_backward)
{
    ReLU relu;
    Matrix z(1, 2);
    z(0, 0) = -2;
    z(0, 1) = 3;

    Matrix a = relu.forward(z, true);
    cr_assert_eq(a(0, 0), 0);
    cr_assert_eq(a(0, 1), 3);
    Matrix dz = relu.backward(Matrix(1, 2, 0.5));
    cr_assert_eq(dz(0, 0), 0);
    cr_assert_eq(dz(0, 1), 0.5);
}

Test(activations, leaky_relu_keeps_a_small_negative_slope)
{
    LeakyReLU leaky(0.01);
    Matrix z(1, 1, -2);

    cr_assert_float_eq(leaky.forward(z, true)(0, 0), -0.02, 1e-12);
    cr_assert_float_eq(leaky.backward(Matrix(1, 1, 1))(0, 0), 0.01, 1e-12);
}

Test(activations, sigmoid_and_tanh_values)
{
    Sigmoid sigmoid;
    Tanh tanh;
    Matrix zero(1, 1);

    cr_assert_float_eq(sigmoid.forward(zero, true)(0, 0), 0.5, 1e-12);
    cr_assert_float_eq(sigmoid.backward(Matrix(1, 1, 1))(0, 0), 0.25, 1e-12);
    cr_assert_float_eq(tanh.forward(zero, true)(0, 0), 0, 1e-12);
    cr_assert_float_eq(tanh.backward(Matrix(1, 1, 1))(0, 0), 1, 1e-12);
}

Test(activations, softmax_rows_are_probabilities)
{
    Softmax softmax;
    Matrix z(1, 3);
    z(0, 0) = 2;
    z(0, 1) = 1;

    Matrix p = softmax.forward(z, true);
    cr_assert_float_eq(p(0, 0), 0.6652409558, 1e-9);
    cr_assert_float_eq(p(0, 1), 0.2447284711, 1e-9);
    cr_assert_float_eq(p(0, 0) + p(0, 1) + p(0, 2), 1, 1e-12);
}

Test(activations, softmax_does_not_overflow)
{
    Softmax softmax;
    Matrix z(1, 2);
    z(0, 0) = 1000;
    z(0, 1) = 999;

    Matrix p = softmax.forward(z, true);
    cr_assert(std::isfinite(p(0, 0)));
    cr_assert_float_eq(p(0, 0), 1 / (1 + std::exp(-1.0)), 1e-12);
}

Test(activations, gradients_match_finite_differences)
{
    ReLU relu;
    LeakyReLU leaky(0.1);
    Sigmoid sigmoid;
    Tanh tanh;
    Softmax softmax;

    cr_assert_lt(inputError(relu, awayFromZero(), 5), tolerance);
    cr_assert_lt(inputError(leaky, awayFromZero(), 5), tolerance);
    cr_assert_lt(inputError(sigmoid, gradient_check::random(4, 5, 12, -3, 3), 5), tolerance);
    cr_assert_lt(inputError(tanh, gradient_check::random(4, 5, 13, -3, 3), 5), tolerance);
    cr_assert_lt(inputError(softmax, gradient_check::random(4, 5, 14, -3, 3), 5), tolerance);
}

Test(activations, have_stable_type_names_and_no_parameters)
{
    ReLU relu;
    LeakyReLU leaky;
    Softmax softmax;

    cr_assert(relu.type() == "relu");
    cr_assert(leaky.type() == "leaky_relu");
    cr_assert(softmax.type() == "softmax");
    cr_assert(relu.parameters().empty());
}
