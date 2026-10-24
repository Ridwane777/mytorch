#include <criterion/criterion.h>

#include <cmath>
#include <memory>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

Matrix xorInputs()
{
    Matrix x(4, 2);
    x(1, 1) = 1;
    x(2, 0) = 1;
    x(3, 0) = 1;
    x(3, 1) = 1;
    return x;
}

// One-hot targets for the two classes "0" and "1".
Matrix xorClasses()
{
    Matrix t(4, 2);
    t(0, 0) = 1;
    t(1, 1) = 1;
    t(2, 1) = 1;
    t(3, 0) = 1;
    return t;
}

Matrix xorScalars()
{
    Matrix t(4, 1);
    t(1, 0) = 1;
    t(2, 0) = 1;
    return t;
}

Network softmaxNet(std::size_t hidden = 8)
{
    Network net;
    net.add(std::make_unique<Dense>(2, hidden, 1));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(hidden, 2, 2));
    net.add(std::make_unique<Softmax>());
    return net;
}

std::vector<Scalar> snapshot(Network &net)
{
    std::vector<Scalar> values;

    for (const Parameter &p : net.parameters())
        values.insert(values.end(), p.value->data(), p.value->data() + p.value->size());
    return values;
}

} // namespace

Test(network, learns_xor_with_softmax_and_cross_entropy)
{
    Network net = softmaxNet();
    CrossEntropy loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 500;
    config.batchSize = 4;

    History history = net.fit({xorInputs(), xorClasses()}, nullptr, loss, adam, config);
    Matrix p = net.predict(xorInputs());
    for (std::size_t r = 0; r < 4; r++) {
        const std::size_t guess = p(r, 1) > p(r, 0) ? 1 : 0;
        cr_assert_eq(guess, xorClasses()(r, 1) == 1 ? 1u : 0u, "row %zu", r);
    }
    cr_assert_lt(history.back().trainLoss, 0.05);
}

Test(network, learns_xor_with_sigmoid_and_mse)
{
    Network net;
    net.add(std::make_unique<Dense>(2, 8, 3));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(8, 1, 4));
    net.add(std::make_unique<Sigmoid>());
    MSE loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 1000;
    config.batchSize = 2;

    net.fit({xorInputs(), xorScalars()}, nullptr, loss, adam, config);
    Matrix p = net.predict(xorInputs());
    for (std::size_t r = 0; r < 4; r++)
        cr_assert_eq(p(r, 0) > 0.5, xorScalars()(r, 0) == 1, "row %zu: %f", r, p(r, 0));
}

Test(network, same_seed_gives_identical_training)
{
    Network a = softmaxNet();
    Network b = softmaxNet();
    CrossEntropy loss;
    Adam optA(0.05);
    Adam optB(0.05);
    TrainConfig config;
    config.epochs = 50;
    config.batchSize = 3;

    History ha = a.fit({xorInputs(), xorClasses()}, nullptr, loss, optA, config);
    History hb = b.fit({xorInputs(), xorClasses()}, nullptr, loss, optB, config);
    for (std::size_t e = 0; e < ha.size(); e++)
        cr_assert_eq(ha[e].trainLoss, hb[e].trainLoss);
    cr_assert(snapshot(a) == snapshot(b));
}

Test(network, another_seed_shuffles_differently)
{
    Network a = softmaxNet();
    Network b = softmaxNet();
    CrossEntropy loss;
    Adam optA(0.05);
    Adam optB(0.05);
    TrainConfig config;
    config.epochs = 5;
    config.batchSize = 1;
    TrainConfig other = config;
    other.seed = 7;

    a.fit({xorInputs(), xorClasses()}, nullptr, loss, optA, config);
    b.fit({xorInputs(), xorClasses()}, nullptr, loss, optB, other);
    cr_assert(snapshot(a) != snapshot(b));
}

Test(network, training_loss_decreases)
{
    Network net = softmaxNet();
    CrossEntropy loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 100;
    config.batchSize = 4;

    History h = net.fit({xorInputs(), xorClasses()}, nullptr, loss, adam, config);
    cr_assert_eq(h.size(), 100);
    // Random weights on two classes start near log(2).
    cr_assert_lt(std::abs(h.front().trainLoss - std::log(2.0)), 0.2);
    cr_assert_lt(h.back().trainLoss, h.front().trainLoss / 4);
}

Test(network, scheduler_sets_the_rate_of_each_epoch)
{
    Network net = softmaxNet();
    CrossEntropy loss;
    SGD sgd(0.5);
    StepDecay schedule(0.5, 2, 0.1);
    TrainConfig config;
    config.epochs = 5;

    net.fit({xorInputs(), xorClasses()}, nullptr, loss, sgd, config, &schedule);
    cr_assert_float_eq(sgd.learningRate(), schedule.rate(4), 1e-15);
}

Test(network, every_example_counts_once_with_a_partial_last_batch)
{
    // 10 examples in batches of 4 (4 + 4 + 2). With a zero rate the weights never move, so the
    // epoch loss must equal the loss of the whole set computed in one go.
    Matrix x(10, 2);
    Matrix t(10, 2);
    for (std::size_t r = 0; r < 10; r++) {
        x(r, 0) = static_cast<Scalar>(r) / 10;
        x(r, 1) = 1 - static_cast<Scalar>(r) / 7;
        t(r, r % 2) = 1;
    }
    Network net = softmaxNet();
    CrossEntropy loss;
    SGD sgd(1);
    sgd.setLearningRate(0);
    TrainConfig config;
    config.epochs = 1;
    config.batchSize = 4;

    History h = net.fit({x, t}, nullptr, loss, sgd, config);
    cr_assert_float_eq(h[0].trainLoss, loss.value(net.predict(x), t), 1e-12);
}

Test(network, fit_backpropagation_matches_finite_differences)
{
    // One full-batch SGD step with rate 1 moves each weight by exactly its gradient, so the
    // gradient used by fit is (before - after). It must match the finite differences of the loss.
    Network net = softmaxNet(3);
    CrossEntropy loss(0.1, {1.0, 2.0});
    const Matrix x = xorInputs();
    const Matrix t = xorClasses();
    std::vector<Scalar> numeric;
    const Scalar h = 1e-6;

    for (const Parameter &p : net.parameters())
        for (std::size_t i = 0; i < p.value->size(); i++) {
            const Scalar saved = p.value->data()[i];
            p.value->data()[i] = saved + h;
            const Scalar plus = loss.value(net.predict(x), t);
            p.value->data()[i] = saved - h;
            const Scalar minus = loss.value(net.predict(x), t);
            p.value->data()[i] = saved;
            numeric.push_back((plus - minus) / (2 * h));
        }
    const std::vector<Scalar> before = snapshot(net);
    SGD sgd(1);
    TrainConfig config;
    config.epochs = 1;
    config.batchSize = 4;
    config.shuffle = false;

    net.fit({x, t}, nullptr, loss, sgd, config);
    const std::vector<Scalar> after = snapshot(net);
    for (std::size_t i = 0; i < numeric.size(); i++) {
        const Scalar analytic = before[i] - after[i];
        const Scalar error = std::abs(analytic - numeric[i]) /
                             std::max(1e-10, std::abs(analytic) + std::abs(numeric[i]));
        cr_assert_lt(error, 1e-6, "weight %zu: %g vs %g", i, analytic, numeric[i]);
    }
}

Test(network, records_validation_loss)
{
    Network net = softmaxNet();
    CrossEntropy loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 3;
    Dataset data{xorInputs(), xorClasses()};

    History h = net.fit(data, &data, loss, adam, config);
    cr_assert_float_eq(h.back().validationLoss, loss.value(net.predict(data.inputs), data.targets),
                       1e-12);
}

Test(network, rejects_invalid_training_setups)
{
    CrossEntropy cross;
    MSE mse;
    Adam adam;
    TrainConfig config;
    Dataset data{xorInputs(), xorClasses()};

    Network noSoftmax;
    noSoftmax.add(std::make_unique<Dense>(2, 2));
    cr_assert_throw(noSoftmax.fit(data, nullptr, cross, adam, config), Error);

    Network net = softmaxNet();
    cr_assert_throw(net.fit({Matrix(0, 2), Matrix(0, 2)}, nullptr, mse, adam, config), Error);
    cr_assert_throw(net.fit({xorInputs(), Matrix(3, 2)}, nullptr, mse, adam, config), Error);
    TrainConfig zero;
    zero.batchSize = 0;
    cr_assert_throw(net.fit(data, nullptr, mse, adam, zero), Error);
    Network empty;
    cr_assert_throw(empty.fit(data, nullptr, mse, adam, config), Error);
    cr_assert_throw(net.add(nullptr), Error);
}
