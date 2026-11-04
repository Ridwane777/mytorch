#include <criterion/criterion.h>

#include <cmath>
#include <memory>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

Matrix spreadWeights(std::size_t rows, std::size_t cols)
{
    // Weights kept away from 0, where L1 is not differentiable.
    return gradient_check::random(rows, cols, 21).map([](Scalar w) {
        return w + (w >= 0 ? 0.1 : -0.1);
    });
}

} // namespace

Test(regularization, penalty_value)
{
    Dense dense(2, 1, 1, {0.5, 0.25});
    dense.weights()(0, 0) = 2;
    dense.weights()(1, 0) = -1;
    dense.bias()(0, 0) = 100; // never penalized

    // 0.5·(2 + 1) + 0.25·(4 + 1)
    cr_assert_float_eq(dense.penalty(), 2.75, 1e-12);
    cr_assert_eq(Dense(2, 1).penalty(), 0);
}

Test(regularization, gradient_includes_the_penalty)
{
    Dense dense(4, 3, 1, {0.03, 0.2});
    dense.weights() = spreadWeights(4, 3);
    Matrix input = gradient_check::random(5, 4, 2);
    const Matrix weights = gradient_check::random(5, 3, 3);
    const Scalar h = gradient_check::step;

    dense.forward(input, true);
    dense.backward(weights);
    const Matrix analytic = *dense.parameters()[0].grad;
    Scalar worst = 0;
    for (std::size_t i = 0; i < dense.weights().size(); i++) {
        Scalar &w = dense.weights().data()[i];
        const Scalar saved = w;
        w = saved + h;
        const Scalar plus = gradient_check::loss(dense, input, weights) + dense.penalty();
        w = saved - h;
        const Scalar minus = gradient_check::loss(dense, input, weights) + dense.penalty();
        w = saved;
        const Scalar numeric = (plus - minus) / (2 * h);
        const Scalar a = analytic.data()[i];
        worst =
            std::max(worst, std::abs(a - numeric) / std::max(1.0, std::abs(a) + std::abs(numeric)));
    }
    cr_assert_lt(worst, gradient_check::tolerance);
}

Test(regularization, bias_gradient_is_not_penalized)
{
    Dense plain(3, 2, 5);
    Dense reg(3, 2, 5, {1, 1});
    Matrix input = gradient_check::random(4, 3, 6);
    Matrix g = gradient_check::random(4, 2, 7);

    plain.forward(input, true);
    plain.backward(g);
    reg.forward(input, true);
    reg.backward(g);
    for (std::size_t i = 0; i < 2; i++)
        cr_assert_eq((*plain.parameters()[1].grad)(0, i), (*reg.parameters()[1].grad)(0, i));
}

Test(regularization, l1_drives_useless_weights_to_zero)
{
    // Only the first input carries the target; L1 must shrink the weight of the second one.
    Matrix x = gradient_check::random(64, 2, 8);
    Matrix y(64, 1);
    for (std::size_t r = 0; r < 64; r++)
        y(r, 0) = 2 * x(r, 0);
    Network net;
    net.add(std::make_unique<Dense>(2, 1, 9, Regularization{0.05, 0}));
    MSE loss;
    SGD sgd(0.05);
    TrainConfig config;
    config.epochs = 400;
    config.batchSize = 64;

    net.fit({x, y}, nullptr, loss, sgd, config);
    const Matrix &w = *net.parameters()[0].value;
    cr_assert_lt(std::abs(w(1, 0)), 0.01, "useless weight %g", w(1, 0));
    cr_assert_gt(std::abs(w(0, 0)), 1.5, "useful weight %g", w(0, 0));
}

Test(regularization, rejects_negative_coefficients)
{
    cr_assert_throw(Dense(2, 2, 1, {-0.1, 0}), Error);
    cr_assert_throw(Dense(2, 2, 1, {0, -0.1}), Error);
}

Test(regularization, fit_reports_the_penalty)
{
    Network net;
    net.add(std::make_unique<Dense>(2, 2, 1, Regularization{0, 0.1}));
    net.add(std::make_unique<Softmax>());
    CrossEntropy loss;
    SGD sgd(1);
    sgd.setLearningRate(0);
    TrainConfig config;
    config.epochs = 1;
    Matrix x = gradient_check::random(4, 2, 3);
    Matrix t(4, 2);
    for (std::size_t r = 0; r < 4; r++)
        t(r, r % 2) = 1;

    History h = net.fit({x, t}, nullptr, loss, sgd, config);
    const auto *dense = dynamic_cast<const Dense *>(net.layers()[0].get());
    cr_assert_float_eq(h[0].penalty, dense->penalty(), 1e-12);
    cr_assert_float_eq(h[0].trainLoss, loss.value(net.predict(x), t), 1e-12);
}

Test(dropout, is_the_identity_in_prediction)
{
    Dropout dropout(0.5);
    Matrix x = gradient_check::random(3, 4, 1);
    Matrix out = dropout.forward(x, false);

    for (std::size_t i = 0; i < x.size(); i++)
        cr_assert_eq(out.data()[i], x.data()[i]);
    Matrix g = gradient_check::random(3, 4, 2);
    Matrix back = dropout.backward(g);
    cr_assert_eq(back.data()[5], g.data()[5]);
}

Test(dropout, zeroes_the_rate_and_keeps_the_mean)
{
    Dropout dropout(0.3, 7);
    Matrix ones(200, 100, 1);
    Matrix out = dropout.forward(ones, true);
    std::size_t zeros = 0;
    Scalar sum = 0;

    for (std::size_t i = 0; i < out.size(); i++) {
        zeros += out.data()[i] == 0;
        sum += out.data()[i];
    }
    const Scalar fraction = static_cast<Scalar>(zeros) / static_cast<Scalar>(out.size());
    cr_assert_lt(std::abs(fraction - 0.3), 0.01, "dropped %f", fraction);
    cr_assert_lt(std::abs(sum / static_cast<Scalar>(out.size()) - 1), 0.02);
}

Test(dropout, backward_reuses_the_mask)
{
    Dropout dropout(0.4, 3);
    Matrix x = gradient_check::random(5, 6, 4);
    Matrix out = dropout.forward(x, true);
    Matrix g(5, 6, 1);
    Matrix back = dropout.backward(g);

    for (std::size_t i = 0; i < x.size(); i++) {
        if (out.data()[i] == 0)
            cr_assert_eq(back.data()[i], 0);
        else
            cr_assert_float_eq(back.data()[i], 1 / 0.6, 1e-12);
    }
}

Test(dropout, same_seed_same_mask)
{
    Dropout a(0.5, 11);
    Dropout b(0.5, 11);
    Matrix x(10, 10, 1);
    Matrix ma = a.forward(x, true);
    Matrix mb = b.forward(x, true);

    for (std::size_t i = 0; i < x.size(); i++)
        cr_assert_eq(ma.data()[i], mb.data()[i]);
}

Test(dropout, rejects_invalid_rates)
{
    cr_assert_throw(Dropout(-0.1), Error);
    cr_assert_throw(Dropout(1), Error);
    cr_assert(Dropout(0).type() == "dropout");
}

namespace {

// Validation loss rises after a few epochs when the network memorizes 8 noisy points.
struct Overfit {
    Dataset train;
    Dataset validation;
};

Overfit overfitting()
{
    Overfit o{{gradient_check::random(8, 2, 1), Matrix(8, 2)},
              {gradient_check::random(64, 2, 2), Matrix(64, 2)}};
    // Labels unrelated to the inputs: anything learned on train is memorization.
    for (std::size_t r = 0; r < 8; r++)
        o.train.targets(r, (r * 5) % 3 == 0) = 1;
    for (std::size_t r = 0; r < 64; r++)
        o.validation.targets(r, (r * 7) % 3 == 0) = 1;
    return o;
}

Network overfitNet()
{
    Network net;
    net.add(std::make_unique<Dense>(2, 32, 1));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(32, 2, 2));
    net.add(std::make_unique<Softmax>());
    return net;
}

std::vector<Scalar> weightsOf(Network &net)
{
    std::vector<Scalar> values;
    for (const Parameter &p : net.parameters())
        values.insert(values.end(), p.value->data(), p.value->data() + p.value->size());
    return values;
}

} // namespace

Test(early_stopping, stops_and_restores_the_best_epoch)
{
    Overfit o = overfitting();
    CrossEntropy loss;
    TrainConfig config;
    config.epochs = 500;
    config.batchSize = 8;
    config.patience = 10;

    Network net = overfitNet();
    Adam adam(0.05);
    History h = net.fit(o.train, &o.validation, loss, adam, config);
    cr_assert_lt(h.size(), 500u, "never stopped");
    std::size_t bestEpoch = 0;
    for (std::size_t e = 1; e < h.size(); e++)
        if (h[e].validationLoss < h[bestEpoch].validationLoss)
            bestEpoch = e;
    cr_assert_eq(h.size(), bestEpoch + 1 + config.patience);
    cr_assert_float_eq(loss.value(net.predict(o.validation.inputs), o.validation.targets),
                       h[bestEpoch].validationLoss, 1e-12);

    // Same run stopped by hand at the best epoch gives the same weights.
    Network again = overfitNet();
    Adam adam2(0.05);
    TrainConfig plain = config;
    plain.patience = 0;
    plain.epochs = bestEpoch + 1;
    again.fit(o.train, &o.validation, loss, adam2, plain);
    cr_assert(weightsOf(net) == weightsOf(again));
}

Test(early_stopping, without_restore_keeps_the_last_weights)
{
    Overfit o = overfitting();
    CrossEntropy loss;
    TrainConfig config;
    config.epochs = 500;
    config.batchSize = 8;
    config.patience = 10;
    config.restoreBest = false;
    Network net = overfitNet();
    Adam adam(0.05);

    History h = net.fit(o.train, &o.validation, loss, adam, config);
    cr_assert_float_eq(loss.value(net.predict(o.validation.inputs), o.validation.targets),
                       h.back().validationLoss, 1e-12);
}

Test(early_stopping, needs_a_validation_set)
{
    Overfit o = overfitting();
    CrossEntropy loss;
    Adam adam;
    TrainConfig config;
    config.patience = 3;
    Network net = overfitNet();

    cr_assert_throw(net.fit(o.train, nullptr, loss, adam, config), Error);
}
