#include <criterion/criterion.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

Matrix row(const std::vector<Scalar> &values)
{
    Matrix m(1, values.size());
    for (std::size_t i = 0; i < values.size(); i++)
        m(0, i) = values[i];
    return m;
}

// 1..9 row-major in one 3 × 3 channel.
Matrix oneToNine()
{
    return row({1, 2, 3, 4, 5, 6, 7, 8, 9});
}

void setWeights(Conv2D &conv, const std::vector<Scalar> &values)
{
    cr_assert_eq(values.size(), conv.weights().size());
    for (std::size_t i = 0; i < values.size(); i++)
        conv.weights().data()[i] = values[i];
}

// Distinct, well separated values (steps of 0.05, far above the finite-difference step), so that
// no maximum is ever close to a tie.
Matrix separated(std::size_t rows, std::size_t cols)
{
    Matrix m(rows, cols);
    for (std::size_t i = 0; i < m.size(); i++)
        m.data()[i] = static_cast<Scalar>((i * 37) % 97) * 0.05;
    return m;
}

} // namespace

// ---- hand-computed forward values --------------------------------------------------------------

Test(conv2d, forward_uses_the_kernel_in_reading_order)
{
    // 3 × 3 input 1..9, 2 × 2 kernel [[1, 2], [3, 4]], bias 0.5: the output (0, 0) is
    // 1·1 + 2·2 + 3·4 + 4·5 + 0.5. A transposed or flipped kernel would give other numbers.
    Conv2D conv({1, 3, 3}, 1, 2, 1, 0);
    setWeights(conv, {1, 2, 3, 4});
    conv.bias()(0, 0) = 0.5;
    const Matrix out = conv.forward(oneToNine(), false);

    cr_assert_eq(out.cols(), 4);
    cr_assert_float_eq(out(0, 0), 37.5, 1e-12);
    cr_assert_float_eq(out(0, 1), 47.5, 1e-12);
    cr_assert_float_eq(out(0, 2), 67.5, 1e-12);
    cr_assert_float_eq(out(0, 3), 77.5, 1e-12);
}

Test(conv2d, forward_with_stride_two_and_padding_one)
{
    // Kernel 1..9 on 1..9, padded to 5 × 5: a 2 × 2 output. Output (0, 0) sees the input
    // rows 0..1 and columns 0..1 through the kernel cells (1, 1), (1, 2), (2, 1), (2, 2).
    Conv2D conv({1, 3, 3}, 1, 3, 2, 1);
    setWeights(conv, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Matrix out = conv.forward(oneToNine(), false);

    cr_assert_eq(out.cols(), 4);
    cr_assert_float_eq(out(0, 0), 1 * 5 + 2 * 6 + 4 * 8 + 5 * 9, 1e-12);
    cr_assert_float_eq(out(0, 1), 2 * 4 + 3 * 5 + 5 * 7 + 6 * 8, 1e-12);
    cr_assert_float_eq(out(0, 2), 4 * 2 + 5 * 3 + 7 * 5 + 8 * 6, 1e-12);
    cr_assert_float_eq(out(0, 3), 5 * 1 + 6 * 2 + 8 * 4 + 9 * 5, 1e-12);
}

Test(conv2d, forward_with_several_channels_and_filters_is_channel_major)
{
    // Two input channels 2 × 2: [1 2 3 4] and [5 6 7 8], kernel 2: one output cell per filter.
    // Filter 0 = channel 0 cell (0, 0) + channel 1 cell (1, 1) = 1 + 8.
    // Filter 1 = 2·(channel 0 cell (0, 1)) + channel 1 cell (1, 0) + bias 10 = 4 + 7 + 10.
    Conv2D conv({2, 2, 2}, 2, 2, 1, 0);
    setWeights(conv, {1, 0, 0, 0, /**/ 0, 0, 0, 1, //
                      0, 2, 0, 0, /**/ 0, 0, 1, 0});
    conv.bias()(0, 1) = 10;
    Matrix input(2, 8);
    for (std::size_t c = 0; c < 8; c++)
        input(0, c) = static_cast<Scalar>(c + 1);
    const Matrix out = conv.forward(input, false); // the second sample is all zeros

    cr_assert_eq(out.cols(), 2);
    cr_assert_float_eq(out(0, 0), 9, 1e-12);
    cr_assert_float_eq(out(0, 1), 21, 1e-12);
    cr_assert_float_eq(out(1, 0), 0, 1e-12);
    cr_assert_float_eq(out(1, 1), 10, 1e-12);
}

Test(conv2d, two_filters_fill_the_output_channel_after_channel)
{
    Conv2D conv({1, 3, 3}, 2, 2, 1, 0);
    setWeights(conv, {1, 2, 3, 4, /**/ 0, 0, 0, 1});
    const Matrix out = conv.forward(oneToNine(), false);
    const std::vector<Scalar> expected = {37, 47, 67, 77, 5, 6, 8, 9};

    cr_assert_eq(out.cols(), 8);
    for (std::size_t i = 0; i < 8; i++)
        cr_assert_float_eq(out(0, i), expected[i], 1e-12, "column %zu", i);
}

Test(conv2d, backward_matches_hand_computation)
{
    // Kernel [[1, 2], [3, 4]] on 1..9; gradient 1 on output (0, 0) and 2 on (1, 1).
    Conv2D conv({1, 3, 3}, 1, 2, 1, 0);
    setWeights(conv, {1, 2, 3, 4});
    conv.forward(oneToNine(), true);
    const Matrix dx = conv.backward(row({1, 0, 0, 2}));
    const std::vector<Scalar> expectedInput = {1, 2, 0, 3, 4 + 2, 4, 0, 6, 8};
    const auto params = conv.parameters();

    for (std::size_t i = 0; i < 9; i++)
        cr_assert_float_eq(dx(0, i), expectedInput[i], 1e-12, "input %zu", i);
    // dW = 1·patch(0, 0) + 2·patch(1, 1) = [1 2 4 5] + 2·[5 6 8 9]
    cr_assert_float_eq((*params[0].grad)(0, 0), 11, 1e-12);
    cr_assert_float_eq((*params[0].grad)(0, 1), 14, 1e-12);
    cr_assert_float_eq((*params[0].grad)(0, 2), 20, 1e-12);
    cr_assert_float_eq((*params[0].grad)(0, 3), 23, 1e-12);
    cr_assert_float_eq((*params[1].grad)(0, 0), 3, 1e-12);
}

Test(conv2d, large_batches_are_processed_by_blocks_with_the_same_result)
{
    Conv2D conv({2, 4, 4}, 3, 3, 1, 1, 4);
    const Matrix input = gradient_check::random(300, 32, 8);
    const Matrix all = conv.forward(input, false);

    for (std::size_t r : {0u, 255u, 256u, 299u}) {
        Matrix one(1, 32);
        for (std::size_t c = 0; c < 32; c++)
            one(0, c) = input(r, c);
        const Matrix alone = conv.forward(one, false);
        for (std::size_t c = 0; c < alone.cols(); c++)
            cr_assert_eq(alone(0, c), all(r, c));
    }
}

Test(max_pool2d, forward_takes_the_maximum_of_each_window_per_channel)
{
    MaxPool2D pool({2, 4, 4}, 2, 2);
    const Matrix input = row({1, 3, 2, 4, 5, 6, 8, 7, 9, 12, 10, 11, 15, 13, 14, 16, //
                              16, 14, 15, 13, 12, 11, 9, 10, 8, 5, 7, 6, 2, 4, 3, 1});
    const Matrix out = pool.forward(input, false);
    const std::vector<Scalar> expected = {6, 8, 15, 16, 16, 15, 8, 7};

    cr_assert_eq(out.cols(), 8);
    for (std::size_t i = 0; i < 8; i++)
        cr_assert_eq(out(0, i), expected[i], "column %zu", i);
}

Test(max_pool2d, rounds_the_output_size_down_and_ties_go_to_the_first_position)
{
    MaxPool2D odd({1, 5, 5}, 2, 2); // 2 × 2: the last row and column are ignored
    cr_assert_eq(odd.outputShape().height, 2);
    cr_assert_eq(odd.outputWidth(25), 4);

    MaxPool2D pool({1, 2, 2}, 2, 2);
    pool.forward(row({1, 1, 1, 1}), false);
    const Matrix dx = pool.backward(row({3}));
    cr_assert_eq(dx(0, 0), 3);
    cr_assert_eq(dx(0, 1), 0);
    cr_assert_eq(dx(0, 2), 0);
    cr_assert_eq(dx(0, 3), 0);
}

Test(max_pool2d, overlapping_windows_add_their_gradients)
{
    // 3 × 3, window 2, stride 1: the centre 9 is the maximum of the four windows.
    MaxPool2D pool({1, 3, 3}, 2, 1);
    const Matrix out = pool.forward(row({1, 2, 3, 4, 9, 6, 7, 8, 5}), false);
    for (std::size_t i = 0; i < 4; i++)
        cr_assert_eq(out(0, i), 9);
    const Matrix dx = pool.backward(row({1, 2, 3, 4}));
    cr_assert_eq(dx(0, 4), 10);
}

Test(avg_pool2d, forward_and_backward_match_hand_computation)
{
    AvgPool2D pool({1, 4, 4}, 2, 2);
    Matrix input(1, 16);
    for (std::size_t i = 0; i < 16; i++)
        input(0, i) = static_cast<Scalar>(i + 1);
    const Matrix out = pool.forward(input, false);

    cr_assert_float_eq(out(0, 0), 3.5, 1e-12);
    cr_assert_float_eq(out(0, 1), 5.5, 1e-12);
    cr_assert_float_eq(out(0, 2), 11.5, 1e-12);
    cr_assert_float_eq(out(0, 3), 13.5, 1e-12);
    const Matrix dx = pool.backward(row({4, 8, 0, 0}));
    cr_assert_float_eq(dx(0, 0), 1, 1e-12);
    cr_assert_float_eq(dx(0, 5), 1, 1e-12);
    cr_assert_float_eq(dx(0, 2), 2, 1e-12);
    cr_assert_float_eq(dx(0, 7), 2, 1e-12);
    cr_assert_float_eq(dx(0, 8), 0, 1e-12);
}

Test(global_avg_pool, forward_and_backward_match_hand_computation)
{
    GlobalAvgPool pool({2, 2, 2});
    const Matrix out = pool.forward(row({1, 2, 3, 4, 5, 6, 7, 8}), false);

    cr_assert_eq(out.cols(), 2);
    cr_assert_float_eq(out(0, 0), 2.5, 1e-12);
    cr_assert_float_eq(out(0, 1), 6.5, 1e-12);
    const Matrix dx = pool.backward(row({4, 8}));
    for (std::size_t i = 0; i < 4; i++) {
        cr_assert_float_eq(dx(0, i), 1, 1e-12);
        cr_assert_float_eq(dx(0, 4 + i), 2, 1e-12);
    }
}

Test(to_planes, forward_maps_the_fen_attacks_901_layout)
{
    BoardPlanes planes(901, 768, 5);
    Matrix input(2, 901);
    for (std::size_t c = 0; c < 901; c++) {
        input(0, c) = static_cast<Scalar>(c);
        input(1, c) = -static_cast<Scalar>(c);
    }
    const Matrix out = planes.forward(input, false);

    cr_assert_eq(planes.outputShape().channels, 19);
    cr_assert_eq(out.cols(), 19 * 64);
    for (std::size_t s = 0; s < 64; s++) {
        cr_assert_eq(out(0, 0 * 64 + s), static_cast<Scalar>(s));             // first piece plane
        cr_assert_eq(out(0, 11 * 64 + s), static_cast<Scalar>(704 + s));      // last piece plane
        cr_assert_eq(out(0, 12 * 64 + s), static_cast<Scalar>(773 + s));      // white attacks
        cr_assert_eq(out(0, 13 * 64 + s), static_cast<Scalar>(837 + s));      // black attacks
        for (std::size_t j = 0; j < 5; j++)
            cr_assert_eq(out(0, (14 + j) * 64 + s), static_cast<Scalar>(768 + j)); // broadcast
        cr_assert_eq(out(1, 13 * 64 + s), -static_cast<Scalar>(837 + s));
    }
}

Test(to_planes, any_plane_geometry_is_accepted)
{
    // Two planes of 3 × 4, then one scalar between them: nothing assumes a chessboard.
    ToPlanes planes(25, 12, 1, 3, 4);
    Matrix input(1, 25);
    for (std::size_t c = 0; c < 25; c++)
        input(0, c) = static_cast<Scalar>(c);
    const Matrix out = planes.forward(input, false);

    cr_assert_eq(planes.outputShape().channels, 3);
    cr_assert_eq(planes.outputShape().height, 3);
    cr_assert_eq(planes.outputShape().width, 4);
    for (std::size_t s = 0; s < 12; s++) {
        cr_assert_eq(out(0, s), static_cast<Scalar>(s));           // first plane
        cr_assert_eq(out(0, 12 + s), static_cast<Scalar>(13 + s)); // second plane, after the scalar
        cr_assert_eq(out(0, 24 + s), 12);                          // the scalar, broadcast
    }
    cr_assert_lt(gradient_check::inputError(planes, gradient_check::random(2, 25, 7),
                                            planes.outputShape().size()),
                 gradient_check::tolerance);
}

Test(to_planes, backward_scatters_squares_and_sums_scalars)
{
    BoardPlanes planes(129, 64, 1); // 64 squares, 1 scalar, 64 squares
    Matrix grad(1, 3 * 64);
    for (std::size_t c = 0; c < grad.cols(); c++)
        grad(0, c) = static_cast<Scalar>(c);
    planes.forward(Matrix(1, 129), false);
    const Matrix dx = planes.backward(grad);
    Scalar sum = 0;

    for (std::size_t s = 0; s < 64; s++) {
        cr_assert_eq(dx(0, s), static_cast<Scalar>(s));
        cr_assert_eq(dx(0, 65 + s), static_cast<Scalar>(64 + s));
        sum += static_cast<Scalar>(128 + s);
    }
    cr_assert_eq(dx(0, 64), sum);
}

// ---- finite-difference gradients ---------------------------------------------------------------

namespace {

struct Geometry {
    PlaneShape input;
    std::size_t filters;
    std::size_t kernel;
    std::size_t stride;
    std::size_t padding;
};

} // namespace

Test(conv2d, gradients_match_finite_differences)
{
    const std::vector<Geometry> geometries = {
        {{2, 5, 5}, 3, 3, 2, 1}, // stride 2 and padding 1, two input channels
        {{1, 4, 4}, 2, 2, 1, 0},
        {{3, 4, 3}, 2, 3, 1, 1}, // not square
        {{2, 4, 4}, 2, 4, 3, 2}, // padding larger than needed, kernel as large as the input
    };
    std::uint32_t seed = 1;
    for (const Geometry &g : geometries) {
        Conv2D conv(g.input, g.filters, g.kernel, g.stride, g.padding, seed++);
        for (std::size_t i = 0; i < conv.bias().size(); i++)
            conv.bias().data()[i] = 0.1 * static_cast<Scalar>(i + 1);
        const Matrix input = gradient_check::random(3, g.input.size(), seed++);
        const std::size_t outputs = conv.outputShape().size();
        const Matrix weights = gradient_check::random(3, outputs, seed++);

        conv.forward(input, true);
        conv.backward(weights);
        for (const auto &p : conv.parameters())
            cr_assert_lt(gradient_check::maxError(conv, input, weights, *p.value, *p.grad),
                         gradient_check::tolerance);
        cr_assert_lt(gradient_check::inputError(conv, input, outputs), gradient_check::tolerance);
    }
}

Test(conv2d, gradients_accumulate_across_backward_calls)
{
    Conv2D conv({1, 3, 3}, 1, 2, 1, 0);
    const Matrix g(1, 4, 1);

    conv.forward(oneToNine(), true);
    conv.backward(g);
    conv.backward(g);
    cr_assert_float_eq((*conv.parameters()[1].grad)(0, 0), 8, 1e-12);
}

Test(max_pool2d, gradients_match_finite_differences)
{
    for (const Geometry &g : std::vector<Geometry>{{{2, 4, 4}, 0, 2, 2, 0}, {{1, 4, 4}, 0, 2, 1, 0},
                                                    {{2, 5, 5}, 0, 3, 2, 0}}) {
        MaxPool2D pool(g.input, g.kernel, g.stride);
        const Matrix input = separated(2, g.input.size());

        cr_assert_lt(
            gradient_check::inputError(pool, input, pool.outputShape().size()),
            gradient_check::tolerance);
    }
}

Test(avg_pool2d, gradients_match_finite_differences)
{
    for (const Geometry &g : std::vector<Geometry>{{{2, 4, 4}, 0, 2, 2, 0}, {{1, 4, 4}, 0, 2, 1, 0},
                                                    {{2, 5, 5}, 0, 3, 2, 0}}) {
        AvgPool2D pool(g.input, g.kernel, g.stride);

        cr_assert_lt(gradient_check::inputError(pool, gradient_check::random(2, g.input.size(), 4),
                                                pool.outputShape().size()),
                     gradient_check::tolerance);
    }
}

Test(global_avg_pool, gradients_match_finite_differences)
{
    GlobalAvgPool pool({3, 3, 2});

    cr_assert_lt(gradient_check::inputError(pool, gradient_check::random(2, 18, 5), 3),
                 gradient_check::tolerance);
}

Test(to_planes, gradients_match_finite_differences)
{
    for (const auto &[inputs, scalars, count] :
         std::vector<std::array<std::size_t, 3>>{{901, 768, 5}, {129, 64, 1}, {132, 0, 4}}) {
        BoardPlanes planes(inputs, scalars, count);

        cr_assert_lt(gradient_check::inputError(planes, gradient_check::random(2, inputs, 6),
                                                planes.outputShape().size()),
                     gradient_check::tolerance);
    }
}

Test(convolution, a_small_cnn_matches_finite_differences_end_to_end)
{
    gradient_check::Layers layers;
    layers.push_back(std::make_unique<ToPlanes>(129, 64, 1, 8, 8));
    layers.push_back(std::make_unique<Conv2D>(PlaneShape{3, 8, 8}, 2, 3, 2, 1, 3));
    layers.push_back(std::make_unique<Tanh>()); // smooth: no kink for the finite differences
    layers.push_back(std::make_unique<AvgPool2D>(PlaneShape{2, 4, 4}, 2, 2));
    layers.push_back(std::make_unique<Dense>(8, 3, 4));
    layers.push_back(std::make_unique<Softmax>());
    Matrix target(2, 3);
    target(0, 1) = 1;
    target(1, 2) = 1;
    CrossEntropy loss;

    cr_assert_lt(
        gradient_check::networkError(layers, loss, gradient_check::random(2, 129, 9), target),
        gradient_check::tolerance);
}

// ---- errors ------------------------------------------------------------------------------------

Test(convolution, layers_reject_out_of_range_geometry)
{
    cr_assert_throw(Conv2D({0, 3, 3}, 1, 2, 1, 0), Error);   // no channel
    cr_assert_throw(Conv2D({1, 3, 3}, 0, 2, 1, 0), Error);   // no filter
    cr_assert_throw(Conv2D({1, 3, 3}, 1, 0, 1, 0), Error);   // empty kernel
    cr_assert_throw(Conv2D({1, 3, 3}, 1, 2, 0, 0), Error);   // stride 0
    cr_assert_throw(Conv2D({1, 3, 3}, 1, 4, 1, 0), Error);   // kernel larger than the input
    cr_assert_no_throw(Conv2D({1, 3, 3}, 1, 5, 1, 1), Error); // ... unless padding makes room
    cr_assert_throw(MaxPool2D({1, 3, 3}, 0, 1), Error);
    cr_assert_throw(MaxPool2D({1, 3, 3}, 2, 0), Error);
    cr_assert_throw(MaxPool2D({1, 3, 3}, 4, 1), Error);
    cr_assert_throw(MaxPool2D({1, 0, 3}, 1, 1), Error);
    cr_assert_throw(AvgPool2D({1, 3, 3}, 0, 1), Error);
    cr_assert_throw(AvgPool2D({1, 3, 3}, 2, 0), Error);
    cr_assert_throw(AvgPool2D({1, 3, 3}, 4, 1), Error);
    cr_assert_throw(GlobalAvgPool({1, 0, 3}), Error);
    cr_assert_throw(ToPlanes(901, 768, 0, 8, 8), Error);   // no scalar
    cr_assert_throw(ToPlanes(901, 700, 5, 8, 8), Error);   // scalars not on a plane boundary
    cr_assert_throw(ToPlanes(901, 768, 6, 8, 8), Error);   // 895 is not a multiple of 64
    cr_assert_throw(ToPlanes(900, 896, 5, 8, 8), Error);   // scalars do not fit
    cr_assert_throw(ToPlanes(5, 0, 5, 8, 8), Error);       // no square plane
    cr_assert_throw(ToPlanes(17, 0, 1, 0, 4), Error);      // empty plane
}

Test(convolution, layers_reject_an_input_of_another_width)
{
    Conv2D conv({1, 3, 3}, 1, 2, 1, 0);
    MaxPool2D max({1, 4, 4}, 2, 2);
    AvgPool2D avg({1, 4, 4}, 2, 2);
    GlobalAvgPool global({1, 4, 4});
    BoardPlanes planes(129, 64, 1);

    cr_assert_throw(conv.forward(Matrix(1, 8), false), Error);
    cr_assert_throw(max.forward(Matrix(1, 15), false), Error);
    cr_assert_throw(avg.forward(Matrix(1, 17), false), Error);
    cr_assert_throw(global.forward(Matrix(1, 4), false), Error);
    cr_assert_throw(planes.forward(Matrix(1, 128), false), Error);
    cr_assert_throw(conv.outputWidth(8), Error);
    cr_assert_eq(conv.outputWidth(9), 4);
    conv.forward(Matrix(2, 9), false);
    cr_assert_throw(conv.backward(Matrix(2, 3)), Error);
}

Test(convolution, initialisation_is_reproducible_and_he_scaled)
{
    Conv2D a({19, 8, 8}, 32, 3, 1, 1, 7);
    Conv2D b({19, 8, 8}, 32, 3, 1, 1, 7);
    Conv2D c({19, 8, 8}, 32, 3, 1, 1, 8);
    double square = 0;
    bool differs = false;

    for (std::size_t i = 0; i < a.weights().size(); i++) {
        cr_assert_eq(a.weights().data()[i], b.weights().data()[i]);
        differs = differs || a.weights().data()[i] != c.weights().data()[i];
        square += a.weights().data()[i] * a.weights().data()[i];
    }
    cr_assert(differs);
    cr_assert_eq(a.weights().rows(), 32);
    cr_assert_eq(a.weights().cols(), 19 * 9);
    // He normal: variance 2 / fan_in with fan_in = 19 · 9.
    cr_assert_lt(std::abs(square / static_cast<double>(a.weights().size()) * (19 * 9) / 2 - 1),
                 0.1);
    cr_assert_eq(a.bias()(0, 0), 0);
}
