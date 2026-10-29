#include <criterion/criterion.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

std::string path(const std::string &base)
{
    return "test_normalization_" + std::to_string(getpid()) + "_" + base;
}

// Gamma and beta away from 1 and 0, so that their gradients are tested in a general case.
template <typename Norm> void spread(Norm &layer, std::uint32_t seed)
{
    layer.gamma() = gradient_check::random(1, layer.features(), seed, 0.5, 2);
    layer.beta() = gradient_check::random(1, layer.features(), seed + 1);
}

template <typename Norm> void checkGradients(Norm &layer, std::size_t rows)
{
    const std::size_t n = layer.features();
    Matrix input = gradient_check::random(rows, n, 2, -3, 3);
    const Matrix weights = gradient_check::random(rows, n, 3);

    cr_assert_lt(gradient_check::inputError(layer, input, n), gradient_check::tolerance);
    for (const Parameter &p : layer.parameters()) {
        p.grad->fill(0);
        layer.forward(input, true);
        layer.backward(weights);
        cr_assert_lt(gradient_check::maxError(layer, input, weights, *p.value, *p.grad),
                     gradient_check::tolerance);
    }
}

Scalar columnMean(const Matrix &m, std::size_t j)
{
    Scalar sum = 0;
    for (std::size_t i = 0; i < m.rows(); i++)
        sum += m(i, j);
    return sum / static_cast<Scalar>(m.rows());
}

Scalar columnVariance(const Matrix &m, std::size_t j)
{
    const Scalar mean = columnMean(m, j);
    Scalar sum = 0;
    for (std::size_t i = 0; i < m.rows(); i++)
        sum += (m(i, j) - mean) * (m(i, j) - mean);
    return sum / static_cast<Scalar>(m.rows());
}

Matrix row(const Matrix &m, std::size_t i)
{
    Matrix out(1, m.cols());
    for (std::size_t j = 0; j < m.cols(); j++)
        out(0, j) = m(i, j);
    return out;
}

Network normalizedNetwork()
{
    Network net;
    net.add(std::make_unique<Dense>(4, 6, 1));
    net.add(std::make_unique<BatchNorm>(6));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dense>(6, 5, 2));
    net.add(std::make_unique<LayerNorm>(5));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(5, 3, 3));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "raw";
    net.info().inputs = 4;
    net.info().labels = {"a", "b", "c"};
    return net;
}

Dataset data()
{
    Matrix x = gradient_check::random(30, 4, 5);
    Matrix t(30, 3);
    for (std::size_t i = 0; i < 30; i++)
        t(i, i % 3) = 1;
    return {x, t};
}

std::vector<Scalar> weights(Network &net)
{
    std::vector<Scalar> all;
    for (const Parameter &p : net.parameters())
        all.insert(all.end(), p.value->data(), p.value->data() + p.value->size());
    return all;
}

} // namespace

Test(normalization, batch_norm_gradients_match_finite_differences)
{
    BatchNorm layer(4);
    spread(layer, 11);
    checkGradients(layer, 6);
}

Test(normalization, layer_norm_gradients_match_finite_differences)
{
    LayerNorm layer(5);
    spread(layer, 13);
    checkGradients(layer, 3);
}

Test(normalization, batch_norm_centres_and_scales_each_column_in_training)
{
    BatchNorm layer(3);
    const Matrix x = gradient_check::random(8, 3, 4, -10, 50);
    const Matrix y = layer.forward(x, true);

    for (std::size_t j = 0; j < 3; j++) {
        cr_assert_float_eq(columnMean(y, j), 0, 1e-12);
        cr_assert_float_eq(columnVariance(y, j), 1, 1e-4); // epsilon makes it slightly below 1
    }
}

Test(normalization, layer_norm_centres_each_row_and_ignores_the_mode)
{
    LayerNorm layer(4);
    spread(layer, 3);
    const Matrix x = gradient_check::random(5, 4, 6, -10, 50);
    const Matrix trained = layer.forward(x, true);
    const Matrix predicted = layer.forward(x, false);

    for (std::size_t i = 0; i < x.size(); i++)
        cr_assert_eq(trained.data()[i], predicted.data()[i]);
    LayerNorm plain(4);
    const Matrix y = plain.forward(x, true);
    for (std::size_t i = 0; i < 5; i++) {
        Scalar sum = 0;
        for (std::size_t j = 0; j < 4; j++)
            sum += y(i, j);
        cr_assert_float_eq(sum, 0, 1e-12);
    }
}

Test(normalization, batch_norm_running_averages_use_the_unbiased_variance)
{
    BatchNorm layer(1, 1); // momentum 1: the running averages are the last batch's
    Matrix x(4, 1);
    for (std::size_t i = 0; i < 4; i++)
        x(i, 0) = static_cast<Scalar>(2 * i + 2); // 2, 4, 6, 8

    layer.forward(x, true);
    cr_assert_float_eq(layer.runningMean()(0, 0), 5, 1e-12);
    cr_assert_float_eq(layer.runningVariance()(0, 0), 20.0 / 3, 1e-12);
}

Test(normalization, batch_norm_predicts_with_the_running_averages)
{
    BatchNorm layer(3);
    spread(layer, 7);
    const Matrix x = gradient_check::random(10, 3, 8, -4, 9);
    for (int i = 0; i < 300; i++) // momentum 0.1: the averages converge to the batch's
        layer.forward(x, true);

    const Matrix batch = layer.forward(x, false);
    for (std::size_t i = 0; i < x.rows(); i++) {
        const Matrix alone = layer.forward(row(x, i), false);
        for (std::size_t j = 0; j < 3; j++)
            cr_assert_eq(alone(0, j), batch(i, j), "a row depends on its batch in prediction");
    }
    // Close to the training output: the running variance is unbiased, the batch one is not.
    const Matrix trained = layer.forward(x, true);
    for (std::size_t i = 0; i < x.size(); i++)
        cr_assert_float_eq(trained.data()[i], batch.data()[i], 0.3);
}

Test(normalization, batch_norm_with_one_row_gives_beta_and_keeps_the_averages)
{
    BatchNorm layer(2);
    spread(layer, 9);
    const Matrix one = gradient_check::random(1, 2, 10);
    const Matrix y = layer.forward(one, true);

    for (std::size_t j = 0; j < 2; j++) {
        cr_assert_float_eq(y(0, j), layer.beta()(0, j), 1e-12);
        cr_assert_eq(layer.runningMean()(0, j), 0);
        cr_assert_eq(layer.runningVariance()(0, j), 1);
    }
    const Matrix grad = layer.backward(gradient_check::random(1, 2, 11));
    for (std::size_t j = 0; j < 2; j++)
        cr_assert_float_eq(grad(0, j), 0, 1e-12);
}

Test(normalization, a_saved_network_predicts_the_same_after_loading)
{
    Network net = normalizedNetwork();
    CrossEntropy loss;
    Adam adam(0.01);
    TrainConfig config;
    config.epochs = 3;
    config.batchSize = 8;
    net.fit(data(), nullptr, loss, adam, config);
    const std::string file = path("net.nn");

    net.save(file);
    Network loaded = Network::load(file);
    std::remove(file.c_str());
    const Matrix a = net.predict(data().inputs);
    const Matrix b = loaded.predict(data().inputs);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i], "prediction %zu differs after loading", i);
}

Test(normalization, read_from_a_configuration_and_width_checked)
{
    const std::string conf = path("net.conf");
    std::ofstream(conf) << "input raw 4\nlabels 2\na\nb\nlayers 6\ndense 4 8\n"
                           "batch_norm 8 momentum=0.2\nrelu\ndense 8 2\nlayer_norm 2\nsoftmax\n";
    Network net = Network::fromConfig(conf);
    const auto *batch = dynamic_cast<const BatchNorm *>(net.layers()[1].get());
    cr_assert(batch != nullptr);
    cr_assert_eq(batch->momentum(), 0.2);

    std::ofstream(conf) << "input raw 4\nlabels 2\na\nb\nlayers 3\ndense 4 8\nbatch_norm 7\n"
                           "dense 8 2\n";
    try {
        Network::fromConfig(conf);
        cr_assert_fail("a batch_norm of the wrong width was accepted");
    } catch (const Error &e) {
        cr_assert(std::string(e.what()).find("normalizes 7 features but receives 8") !=
                      std::string::npos,
                  "got '%s'", e.what());
    }
    std::remove(conf.c_str());
}

Test(normalization, a_resumed_training_keeps_the_running_averages)
{
    const std::string file = path("resume.ckpt");
    CrossEntropy loss;
    TrainConfig config;
    config.batchSize = 8;

    Network continuous = normalizedNetwork();
    Adam adam(0.01);
    config.epochs = 4;
    continuous.fit(data(), nullptr, loss, adam, config);

    Network first = normalizedNetwork();
    Adam firstAdam(0.01);
    config.epochs = 2;
    first.fit(data(), nullptr, loss, firstAdam, config, nullptr, file);
    Network resumed = normalizedNetwork();
    Adam resumedAdam(0.01);
    config.epochs = 4;
    resumed.fit(data(), nullptr, loss, resumedAdam, config, nullptr, file);
    std::remove(file.c_str());

    cr_assert(weights(resumed) == weights(continuous));
    const Matrix a = continuous.predict(data().inputs);
    const Matrix b = resumed.predict(data().inputs);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i], "the running averages were not restored");
}

Test(normalization, rejects_invalid_settings)
{
    cr_assert_throw(BatchNorm(0), Error);
    cr_assert_throw(BatchNorm(3, 0), Error);
    cr_assert_throw(BatchNorm(3, 1.5), Error);
    cr_assert_throw(BatchNorm(3, 0.1, 0), Error);
    cr_assert_throw(LayerNorm(0), Error);
    cr_assert_throw(LayerNorm(3, -1), Error);
    BatchNorm layer(3);
    cr_assert_throw(layer.forward(Matrix(2, 4), true), Error);
    layer.forward(Matrix(2, 3), false);
    cr_assert_throw(layer.backward(Matrix(2, 3)), Error);
}
