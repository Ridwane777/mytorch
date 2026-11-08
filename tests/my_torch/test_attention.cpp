#include <criterion/criterion.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

// Every gradient below is compared with centred finite differences (gradient_check.hpp, step 1e-6)
// in double precision; the tolerance is gradient_check::tolerance = 1e-6 on the relative error
// (absolute below 1, see maxError). The whole-network check uses networkError, whose relative
// error has a floor of 1e-10 by default; it is raised to 1e-3 there (see the test) because the bias
// of the keys has a gradient of exactly 0.

namespace {

Matrix row(const std::vector<Scalar> &values)
{
    Matrix m(1, values.size());
    for (std::size_t i = 0; i < values.size(); i++)
        m(0, i) = values[i];
    return m;
}

void fillRandom(Matrix &m, std::uint32_t seed)
{
    m = gradient_check::random(m.rows(), m.cols(), seed);
}

// Random projections, so that the attention is far from uniform.
void randomize(SelfAttention &attention, std::uint32_t seed)
{
    for (std::size_t i = 0; i < 4; i++) {
        fillRandom(attention.weights(i), seed + 2 * static_cast<std::uint32_t>(i));
        fillRandom(attention.bias(i), seed + 2 * static_cast<std::uint32_t>(i) + 1);
    }
}

// Checks dX and the gradient of every parameter of the layer for a random batch.
void checkGradients(Layer &layer, std::size_t rows, std::size_t inputWidth, std::size_t outputWidth,
                    std::uint32_t seed, Scalar tolerance = gradient_check::tolerance)
{
    const Matrix input = gradient_check::random(rows, inputWidth, seed);
    const Matrix weights = gradient_check::random(rows, outputWidth, seed + 1);

    for (const Parameter &p : layer.parameters())
        p.grad->fill(0);
    layer.forward(input, true);
    layer.backward(weights);
    for (const Parameter &p : layer.parameters())
        cr_assert_lt(gradient_check::maxError(layer, input, weights, *p.value, *p.grad), tolerance);
    cr_assert_lt(gradient_check::inputError(layer, input, outputWidth, seed + 2), tolerance);
}

// 8 inputs: planes a, b (2 cells each), the scalars s0, s1, then the plane c; 2 tokens.
Matrix tokensInput()
{
    return row({10, 11, 20, 21, 1, 2, 30, 31});
}

} // namespace

// ---- to_tokens ---------------------------------------------------------------------------------

Test(to_tokens, hand_built_input_gives_the_known_tokens)
{
    ToTokens layer(8, 4, 2, 2);
    cr_assert_eq(layer.features(), 5u);
    cr_assert_eq(layer.outputWidth(8), 10u);
    const Matrix out = layer.forward(tokensInput(), false);

    // Token 0: cell 0 of every plane, then the scalars; token 1: cell 1 of every plane.
    const std::vector<Scalar> expected = {10, 20, 30, 1, 2, 11, 21, 31, 1, 2};
    cr_assert_eq(out.cols(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++)
        cr_assert_eq(out(0, i), expected[i], "column %zu", i);
}

Test(to_tokens, backward_sums_the_scalars_over_the_tokens)
{
    ToTokens layer(8, 4, 2, 2);
    layer.forward(tokensInput(), false);
    Matrix g(1, 10);
    for (std::size_t i = 0; i < 10; i++)
        g(0, i) = static_cast<Scalar>(i + 1);
    const Matrix grad = layer.backward(g);

    // Cells: plane a gets (token 0, token 1) = (g0, g5), plane b (g1, g6), plane c (g2, g7).
    cr_assert_eq(grad.cols(), 8u);
    const std::vector<Scalar> expected = {1, 6, 2, 7, 4 + 9, 5 + 10, 3, 8};
    for (std::size_t i = 0; i < expected.size(); i++)
        cr_assert_eq(grad(0, i), expected[i], "column %zu", i);
}

Test(to_tokens, board_layout_maps_squares_to_tokens)
{
    // 12 piece planes, 5 scalars, 2 attack planes of 64 squares: 64 tokens of 19 features.
    ToTokens layer(901, 768, 5, 64);
    cr_assert_eq(layer.features(), 19u);
    cr_assert_eq(layer.outputWidth(901), 64u * 19u);
    Matrix input(1, 901);
    input(0, 5 * 64 + 3) = 1; // piece plane 5, square 3
    input(0, 768 + 2) = 7;    // the third scalar
    input(0, 773 + 9) = 1;    // attack plane 0, square 9
    input(0, 837 + 63) = 1;   // attack plane 1, square 63
    const Matrix out = layer.forward(input, false);
    cr_assert_eq(out(0, 3 * 19 + 5), 1);
    cr_assert_eq(out(0, 9 * 19 + 12), 1);
    cr_assert_eq(out(0, 63 * 19 + 13), 1);
    for (std::size_t t = 0; t < 64; t++)
        cr_assert_eq(out(0, t * 19 + 14 + 2), 7);
    Scalar total = 0;
    for (std::size_t i = 0; i < out.size(); i++)
        total += out.data()[i];
    cr_assert_eq(total, 1 + 1 + 1 + 64 * 7);
}

Test(to_tokens, gradient_matches_finite_differences)
{
    ToTokens layer(8, 4, 2, 2);
    cr_assert_lt(gradient_check::inputError(layer, gradient_check::random(3, 8, 1), 10),
                 gradient_check::tolerance);
    ToTokens first(11, 0, 3, 4); // scalars before every plane: 4 tokens of 5 features
    cr_assert_lt(gradient_check::inputError(first, gradient_check::random(2, 11, 2), 20),
                 gradient_check::tolerance);
}

Test(to_tokens, refuses_a_bad_layout)
{
    cr_assert_throw(ToTokens(8, 4, 2, 0), Error); // no token
    cr_assert_throw(ToTokens(8, 4, 0, 2), Error); // no scalar
    cr_assert_throw(ToTokens(8, 3, 2, 2), Error); // scalars not on a plane boundary
    cr_assert_throw(ToTokens(8, 8, 2, 2), Error); // scalars do not fit
    cr_assert_throw(ToTokens(9, 4, 2, 2), Error); // inputs - count not a multiple of tokens
    cr_assert_throw(ToTokens(2, 0, 2, 2), Error); // no plane at all
    ToTokens layer(8, 4, 2, 2);
    cr_assert_throw(layer.forward(Matrix(1, 7), false), Error);
    cr_assert_throw(layer.outputWidth(9), Error);
    layer.forward(tokensInput(), false);
    cr_assert_throw(layer.backward(Matrix(1, 9)), Error);
}

// ---- token_dense -------------------------------------------------------------------------------

Test(token_dense, applies_the_same_dense_layer_to_every_token)
{
    TokenDense layer(3, 4, 5, 9);
    Dense reference(4, 5, 9, {}, Init::HeNormal);
    fillRandom(layer.bias(), 3);
    reference.bias() = layer.bias();
    const Matrix x = gradient_check::random(2, 12, 1);
    const Matrix out = layer.forward(x, false);

    cr_assert_eq(out.cols(), 15u);
    for (std::size_t r = 0; r < 2; r++)
        for (std::size_t t = 0; t < 3; t++) {
            Matrix token(1, 4);
            for (std::size_t c = 0; c < 4; c++)
                token(0, c) = x(r, t * 4 + c);
            const Matrix expected = reference.forward(token, false);
            for (std::size_t c = 0; c < 5; c++)
                cr_assert_float_eq(out(r, t * 5 + c), expected(0, c), 1e-14);
        }
}

Test(token_dense, gradients_match_finite_differences)
{
    TokenDense layer(3, 4, 5, 2);
    fillRandom(layer.bias(), 3);
    checkGradients(layer, 3, 12, 15, 10);
}

Test(token_dense, penalty_is_the_one_of_dense)
{
    TokenDense plain(3, 4, 2, 5);
    TokenDense penalized(3, 4, 2, 5, Regularization{0.5, 0.25});
    cr_assert_eq(plain.penalty(), 0);
    Scalar expected = 0;
    for (std::size_t i = 0; i < plain.weights().size(); i++) {
        const Scalar w = plain.weights().data()[i];
        expected += 0.5 * std::abs(w) + 0.25 * w * w;
    }
    cr_assert_float_eq(penalized.penalty(), expected, 1e-14);

    const Matrix x = gradient_check::random(2, 12, 4);
    const Matrix g = gradient_check::random(2, 6, 5);
    plain.forward(x, true);
    plain.backward(g);
    penalized.forward(x, true);
    penalized.backward(g);
    for (std::size_t i = 0; i < plain.weights().size(); i++) {
        const Scalar w = plain.weights().data()[i];
        const Scalar sign = w > 0 ? 1 : (w < 0 ? -1 : 0);
        cr_assert_float_eq((*penalized.parameters()[0].grad).data()[i] -
                               (*plain.parameters()[0].grad).data()[i],
                           0.5 * sign + 2 * 0.25 * w, 1e-13);
    }
    cr_assert_throw(TokenDense(3, 4, 2, 5, Regularization{-1, 0}), Error);
}

Test(token_dense, refuses_bad_sizes_and_widths)
{
    cr_assert_throw(TokenDense(0, 4, 2), Error);
    cr_assert_throw(TokenDense(3, 0, 2), Error);
    cr_assert_throw(TokenDense(3, 4, 0), Error);
    TokenDense layer(3, 4, 2);
    cr_assert_eq(layer.outputWidth(12), 6u);
    cr_assert_throw(layer.outputWidth(8), Error);
    cr_assert_throw(layer.forward(Matrix(1, 8), false), Error);
    layer.forward(Matrix(2, 12), false);
    cr_assert_throw(layer.backward(Matrix(2, 5)), Error);
    cr_assert_throw(layer.backward(Matrix(3, 6)), Error);
}

// ---- positional --------------------------------------------------------------------------------

Test(positional, adds_the_table_to_every_sample)
{
    Positional layer(3, 2, 4);
    const Matrix x = gradient_check::random(4, 6, 1);
    const Matrix out = layer.forward(x, false);
    for (std::size_t r = 0; r < 4; r++)
        for (std::size_t t = 0; t < 3; t++)
            for (std::size_t d = 0; d < 2; d++)
                cr_assert_eq(out(r, t * 2 + d), x(r, t * 2 + d) + layer.table()(t, d));
}

Test(positional, table_is_small_and_depends_on_the_seed)
{
    Positional a(64, 32, 1);
    Positional b(64, 32, 1);
    Positional c(64, 32, 2);
    Scalar squares = 0;
    bool different = false;
    for (std::size_t i = 0; i < a.table().size(); i++) {
        cr_assert_eq(a.table().data()[i], b.table().data()[i]);
        different = different || a.table().data()[i] != c.table().data()[i];
        squares += a.table().data()[i] * a.table().data()[i];
    }
    cr_assert(different);
    const Scalar deviation = std::sqrt(squares / static_cast<Scalar>(a.table().size()));
    cr_assert_gt(deviation, 0.08);
    cr_assert_lt(deviation, 0.12);
}

Test(positional, gradients_match_finite_differences)
{
    Positional layer(3, 4, 6);
    checkGradients(layer, 3, 12, 12, 20);
}

Test(positional, refuses_bad_sizes_and_widths)
{
    cr_assert_throw(Positional(0, 4), Error);
    cr_assert_throw(Positional(3, 0), Error);
    Positional layer(3, 4);
    cr_assert_throw(layer.forward(Matrix(1, 11), false), Error);
    cr_assert_throw(layer.outputWidth(13), Error);
    cr_assert_throw(layer.backward(Matrix(1, 13)), Error);
}

// ---- token_norm --------------------------------------------------------------------------------

Test(token_norm, normalizes_each_token_on_its_own)
{
    TokenNorm layer(3, 4);
    const Matrix x = gradient_check::random(2, 12, 1, -3, 3);
    const Matrix out = layer.forward(x, false);
    for (std::size_t r = 0; r < 2; r++)
        for (std::size_t t = 0; t < 3; t++) {
            Scalar mean = 0;
            Scalar squares = 0;
            for (std::size_t d = 0; d < 4; d++)
                mean += out(r, t * 4 + d) / 4;
            for (std::size_t d = 0; d < 4; d++)
                squares += (out(r, t * 4 + d) - mean) * (out(r, t * 4 + d) - mean) / 4;
            cr_assert_float_eq(mean, 0, 1e-12);
            cr_assert_float_eq(squares, 1, 1e-3);
        }
}

Test(token_norm, matches_layer_norm_on_each_token)
{
    TokenNorm layer(2, 3, 1e-4);
    LayerNorm reference(3, 1e-4);
    const Matrix x = gradient_check::random(2, 6, 2);
    const Matrix out = layer.forward(x, false);
    for (std::size_t r = 0; r < 2; r++)
        for (std::size_t t = 0; t < 2; t++) {
            Matrix token(1, 3);
            for (std::size_t d = 0; d < 3; d++)
                token(0, d) = x(r, t * 3 + d);
            const Matrix expected = reference.forward(token, false);
            for (std::size_t d = 0; d < 3; d++)
                cr_assert_eq(out(r, t * 3 + d), expected(0, d));
        }
}

Test(token_norm, gradients_match_finite_differences)
{
    TokenNorm layer(3, 4);
    fillRandom(layer.gamma(), 3);
    fillRandom(layer.beta(), 4);
    checkGradients(layer, 3, 12, 12, 30);
}

Test(token_norm, refuses_bad_arguments_and_widths)
{
    cr_assert_throw(TokenNorm(0, 4), Error);
    cr_assert_throw(TokenNorm(3, 0), Error);
    cr_assert_throw(TokenNorm(3, 4, 0), Error);
    cr_assert_throw(TokenNorm(3, 4, -1e-5), Error);
    TokenNorm layer(3, 4);
    cr_assert_throw(layer.forward(Matrix(1, 11), false), Error);
    cr_assert_throw(layer.outputWidth(11), Error);
    cr_assert_throw(layer.backward(Matrix(1, 11)), Error);
}

// ---- self_attention ----------------------------------------------------------------------------

Test(self_attention, uniform_attention_averages_the_tokens)
{
    // Wq = Wk = 0 makes every score 0 and every attention row 1 / tokens; with Wv = Wo = I and
    // zero biases the output of each token is the mean of the tokens.
    SelfAttention layer(3, 2, 1);
    layer.weights(0).fill(0);
    layer.weights(1).fill(0);
    for (std::size_t i = 2; i < 4; i++) {
        layer.weights(i).fill(0);
        layer.weights(i)(0, 0) = 1;
        layer.weights(i)(1, 1) = 1;
    }
    const Matrix x = row({1, 2, 3, 4, 8, 12});
    const Matrix out = layer.forward(x, true);
    for (std::size_t t = 0; t < 3; t++) {
        cr_assert_float_eq(out(0, t * 2), 4, 1e-14);
        cr_assert_float_eq(out(0, t * 2 + 1), 6, 1e-14);
    }
    for (std::size_t i = 0; i < layer.attention().size(); i++)
        cr_assert_float_eq(layer.attention().data()[i], 1.0 / 3, 1e-14);
}

Test(self_attention, one_hot_attention_copies_the_chosen_token)
{
    // Every query is (30, 0) (Wq = 0, bq = (30, 0)) and a key is (x0, 0), so the score of a key is
    // 30·x0 / sqrt(2): the key with the larger first coordinate takes the whole softmax, and with
    // Wv = Wo = I every token outputs that token.
    SelfAttention layer(2, 2, 1);
    layer.weights(0).fill(0);
    layer.bias(0)(0, 0) = 30; // every query is (30, 0)
    layer.weights(1).fill(0);
    layer.weights(1)(0, 0) = 1; // key = (x0, 0)
    for (std::size_t i = 2; i < 4; i++) {
        layer.weights(i).fill(0);
        layer.weights(i)(0, 0) = 1;
        layer.weights(i)(1, 1) = 1;
    }
    // Token 1 has the larger first coordinate: both tokens attend to it.
    const Matrix out = layer.forward(row({1, 5, 2, 7}), false);
    for (std::size_t t = 0; t < 2; t++) {
        cr_assert_float_eq(out(0, t * 2), 2, 1e-6);
        cr_assert_float_eq(out(0, t * 2 + 1), 7, 1e-6);
    }
}

Test(self_attention, gradients_match_finite_differences)
{
    for (const std::size_t heads : {1u, 2u, 4u}) {
        SelfAttention layer(4, 4, heads, 1);
        randomize(layer, 100);
        checkGradients(layer, 3, 16, 16, 40 + static_cast<std::uint32_t>(heads));
    }
}

Test(self_attention, gradients_match_finite_differences_with_one_token_and_wide_heads)
{
    SelfAttention single(1, 4, 2, 3);
    randomize(single, 200);
    checkGradients(single, 2, 4, 4, 50);
    SelfAttention wide(3, 6, 3, 4);
    randomize(wide, 300);
    checkGradients(wide, 2, 18, 18, 60);
}

Test(self_attention, the_bias_of_the_keys_has_no_gradient)
{
    // Adding the same vector to every key shifts all the scores of a query by the same amount,
    // which a softmax ignores: dbk is exactly 0 in theory.
    SelfAttention layer(4, 4, 2, 1);
    randomize(layer, 100);
    layer.forward(gradient_check::random(3, 16, 1), true);
    layer.backward(gradient_check::random(3, 16, 2));
    const Matrix &dbk = *layer.parameters()[3].grad;
    for (std::size_t i = 0; i < dbk.size(); i++)
        cr_assert_float_eq(dbk.data()[i], 0, 1e-12);
    cr_assert_gt(std::abs((*layer.parameters()[1].grad)(0, 0)), 1e-6); // dbq does not vanish
}

Test(self_attention, attention_rows_sum_to_one)
{
    SelfAttention layer(5, 4, 2, 1);
    randomize(layer, 100);
    layer.forward(gradient_check::random(3, 20, 1, -2, 2), true);
    const Matrix &p = layer.attention();
    cr_assert_eq(p.rows(), 3u * 2u * 5u);
    cr_assert_eq(p.cols(), 5u);
    for (std::size_t r = 0; r < p.rows(); r++) {
        Scalar sum = 0;
        for (std::size_t c = 0; c < p.cols(); c++) {
            cr_assert_geq(p(r, c), 0);
            sum += p(r, c);
        }
        cr_assert_float_eq(sum, 1, 1e-14, "row %zu", r);
    }
}

Test(self_attention, output_does_not_depend_on_the_batch)
{
    SelfAttention layer(4, 4, 2, 1);
    randomize(layer, 100);
    const Matrix x = gradient_check::random(5, 16, 3);
    const Matrix together = layer.forward(x, false);
    for (std::size_t r = 0; r < 5; r++) {
        Matrix single(1, 16);
        for (std::size_t c = 0; c < 16; c++)
            single(0, c) = x(r, c);
        const Matrix alone = layer.forward(single, false);
        for (std::size_t c = 0; c < 16; c++)
            cr_assert_eq(alone(0, c), together(r, c));
    }
}

Test(self_attention, training_flag_changes_nothing_but_the_kept_attention)
{
    SelfAttention layer(4, 4, 2, 1);
    randomize(layer, 100);
    const Matrix x = gradient_check::random(3, 16, 3);
    const Matrix a = layer.forward(x, true);
    cr_assert_gt(layer.attention().size(), 0u);
    const Matrix b = layer.forward(x, false);
    cr_assert_eq(layer.attention().size(), 0u);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i]);
}

Test(self_attention, backward_after_a_prediction_forward_is_the_same)
{
    // Saliency (ADR-0035) calls backward after a prediction forward, which keeps no attention
    // matrix: backward recomputes them and must give the very same gradients.
    SelfAttention trained(4, 4, 2, 1);
    SelfAttention predicted(4, 4, 2, 1);
    randomize(trained, 100);
    randomize(predicted, 100);
    const Matrix x = gradient_check::random(3, 16, 3);
    const Matrix g = gradient_check::random(3, 16, 4);
    trained.forward(x, true);
    predicted.forward(x, false);
    const Matrix dx = trained.backward(g);
    const Matrix dy = predicted.backward(g);
    for (std::size_t i = 0; i < dx.size(); i++)
        cr_assert_eq(dx.data()[i], dy.data()[i]);
    const auto a = trained.parameters();
    const auto b = predicted.parameters();
    for (std::size_t p = 0; p < a.size(); p++)
        for (std::size_t i = 0; i < a[p].grad->size(); i++)
            cr_assert_eq(a[p].grad->data()[i], b[p].grad->data()[i]);
}

Test(self_attention, without_positions_it_is_equivariant_to_a_permutation_of_the_tokens)
{
    SelfAttention layer(4, 4, 2, 1);
    randomize(layer, 100);
    const Matrix x = gradient_check::random(1, 16, 3);
    Matrix swapped = x;
    for (std::size_t d = 0; d < 4; d++)
        std::swap(swapped(0, 0 * 4 + d), swapped(0, 2 * 4 + d));
    const Matrix out = layer.forward(x, false);
    const Matrix swappedOut = layer.forward(swapped, false);
    for (std::size_t d = 0; d < 4; d++) {
        cr_assert_float_eq(out(0, 0 * 4 + d), swappedOut(0, 2 * 4 + d), 1e-12);
        cr_assert_float_eq(out(0, 2 * 4 + d), swappedOut(0, 0 * 4 + d), 1e-12);
        cr_assert_float_eq(out(0, 1 * 4 + d), swappedOut(0, 1 * 4 + d), 1e-12);
    }
}

Test(self_attention, gradients_accumulate_across_backward_calls)
{
    SelfAttention layer(3, 4, 2, 1);
    randomize(layer, 100);
    const Matrix x = gradient_check::random(2, 12, 3);
    const Matrix g = gradient_check::random(2, 12, 4);
    layer.forward(x, true);
    layer.backward(g);
    const Scalar once = (*layer.parameters()[0].grad)(0, 0);
    layer.backward(g);
    cr_assert_float_eq((*layer.parameters()[0].grad)(0, 0), 2 * once, 1e-12);
}

Test(self_attention, refuses_bad_arguments_and_widths)
{
    cr_assert_throw(SelfAttention(4, 6, 4), Error); // 6 is not a multiple of 4
    cr_assert_throw(SelfAttention(0, 4, 2), Error);
    cr_assert_throw(SelfAttention(4, 0, 2), Error);
    cr_assert_throw(SelfAttention(4, 4, 0), Error);
    SelfAttention layer(4, 4, 2);
    cr_assert_eq(layer.outputWidth(16), 16u);
    cr_assert_throw(layer.outputWidth(15), Error);
    cr_assert_throw(layer.forward(Matrix(1, 15), false), Error);
    layer.forward(Matrix(2, 16), true);
    cr_assert_throw(layer.backward(Matrix(2, 15)), Error);
    cr_assert_throw(layer.backward(Matrix(3, 16)), Error);
    cr_assert_throw(layer.weights(4), std::out_of_range);
}

// ---- a whole block -----------------------------------------------------------------------------

namespace {

LayerSequence attentionBranch(std::size_t tokens, std::size_t dim, std::size_t heads,
                              std::uint32_t seed)
{
    LayerSequence layers;
    layers.push_back(std::make_unique<TokenNorm>(tokens, dim));
    auto attention = std::make_unique<SelfAttention>(tokens, dim, heads, seed);
    randomize(*attention, seed * 10);
    layers.push_back(std::move(attention));
    return layers;
}

LayerSequence feedForwardBranch(std::size_t tokens, std::size_t dim, std::uint32_t seed)
{
    LayerSequence layers;
    layers.push_back(std::make_unique<TokenNorm>(tokens, dim));
    layers.push_back(std::make_unique<TokenDense>(tokens, dim, 4 * dim, seed));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<TokenDense>(tokens, 4 * dim, dim, seed + 1));
    return layers;
}

// The pre-norm block of ADR-0047, built from the residual container.
void addBlock(gradient_check::Layers &layers, std::size_t tokens, std::size_t dim,
              std::size_t heads, std::uint32_t seed)
{
    layers.push_back(std::make_unique<Residual>(attentionBranch(tokens, dim, heads, seed)));
    layers.push_back(std::make_unique<Residual>(feedForwardBranch(tokens, dim, seed + 5)));
}

} // namespace

Test(transformer_block, residual_block_gradients_match_finite_differences)
{
    // The two residual layers of the block, one at a time.
    Residual attention(attentionBranch(4, 4, 2, 1));
    checkGradients(attention, 3, 16, 16, 70);
    Residual feedForward(feedForwardBranch(4, 4, 6));
    checkGradients(feedForward, 3, 16, 16, 80);
}

Test(transformer_block, whole_network_gradients_match_finite_differences)
{
    // 9 inputs -> 4 tokens of 3 features -> 4 -> positions -> two blocks -> head -> 3 classes.
    gradient_check::Layers layers;
    layers.push_back(std::make_unique<ToTokens>(9, 8, 1, 4));
    layers.push_back(std::make_unique<TokenDense>(4, 3, 4, 1));
    layers.push_back(std::make_unique<Positional>(4, 4, 2));
    addBlock(layers, 4, 4, 2, 3);
    addBlock(layers, 4, 4, 1, 13);
    layers.push_back(std::make_unique<Dense>(16, 6, 4));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Dense>(6, 3, 5));
    layers.push_back(std::make_unique<Softmax>());
    Matrix target(6, 3);
    for (std::size_t r = 0; r < 6; r++)
        target(r, r % 3) = 1;
    // The bias of the keys has a gradient of exactly 0 (a softmax ignores a shift of all the
    // scores of a row): the floor of 1e-3 on the denominator compares such gradients in absolute
    // terms, instead of against the rounding noise (about 1e-11) of the finite difference.
    cr_assert_lt(gradient_check::networkError(layers, CrossEntropy(),
                                              gradient_check::random(6, 9, 7), target, 1e-3),
                 gradient_check::tolerance);
}
