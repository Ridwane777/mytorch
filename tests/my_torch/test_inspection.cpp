#include <criterion/criterion.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Dense 4 -> 6, BatchNorm, Tanh (smooth, unlike ReLU, for finite differences), Dropout, Dense 6 ->
// 3, Softmax; trained a little so that the BatchNorm running averages are not their initial values.
Network network()
{
    Network net;
    net.add(std::make_unique<Dense>(4, 6, 1));
    net.add(std::make_unique<BatchNorm>(6));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dropout>(0.3, 2));
    net.add(std::make_unique<Dense>(6, 3, 3));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "raw";
    net.info().inputs = 4;
    net.info().labels = {"a", "b", "c"};
    Matrix x = gradient_check::random(24, 4, 3);
    Matrix t(24, 3);
    for (std::size_t i = 0; i < 24; i++)
        t(i, i % 3) = 1;
    CrossEntropy loss;
    Adam adam(0.01);
    TrainConfig config;
    config.epochs = 3;
    config.batchSize = 8;
    net.fit({x, t}, nullptr, loss, adam, config);
    return net;
}

} // namespace

Test(inspection, input_gradient_matches_finite_differences)
{
    Network net = network();
    Matrix x = gradient_check::random(3, 4, 5);
    const Scalar h = 1e-6;

    for (std::size_t output = 0; output < 3; output++) {
        const Matrix analytic = net.inputGradient(x, output);
        for (std::size_t i = 0; i < x.size(); i++) {
            const std::size_t row = i / 4;
            const Scalar saved = x.data()[i];
            x.data()[i] = saved + h;
            const Scalar plus = net.predict(x)(row, output);
            x.data()[i] = saved - h;
            const Scalar minus = net.predict(x)(row, output);
            x.data()[i] = saved;
            const Scalar numeric = (plus - minus) / (2 * h);
            cr_assert_float_eq(analytic.data()[i], numeric, 1e-7, "output %zu, input %zu", output,
                               i);
        }
    }
}

Test(inspection, input_gradient_leaves_the_parameter_gradients)
{
    Network net = network();
    std::vector<Matrix> before;
    for (const Parameter &p : net.parameters())
        before.push_back(*p.grad);

    net.inputGradient(gradient_check::random(2, 4, 6), 1);
    const std::vector<Parameter> after = net.parameters();
    for (std::size_t i = 0; i < after.size(); i++)
        for (std::size_t k = 0; k < before[i].size(); k++)
            cr_assert_eq(after[i].grad->data()[k], before[i].data()[k]);
    cr_assert_throw(net.inputGradient(gradient_check::random(2, 4, 6), 3), Error);
}

Test(inspection, summary_lists_every_layer_and_counts_the_parameters)
{
    Network net = network();
    const std::string text = modelSummary(net);

    // 4·6 + 6, 2·6 (gamma, beta), 6·3 + 3.
    cr_assert(text.find("total parameters: 63") != std::string::npos, "%s", text.c_str());
    for (const char *type : {"dense", "batch_norm", "tanh", "dropout", "softmax"})
        cr_assert(text.find(type) != std::string::npos, "%s missing:\n%s", type, text.c_str());
    cr_assert(text.find("5   dense               3          21") != std::string::npos, "%s",
              text.c_str());
}

Test(inspection, on_epoch_is_called_after_every_epoch)
{
    Network net = network();
    std::vector<std::size_t> epochs;
    std::vector<Scalar> losses;
    TrainConfig config;
    config.epochs = 4;
    config.batchSize = 8;
    config.onEpoch = [&](std::size_t epoch, const EpochStats &stats) {
        epochs.push_back(epoch);
        losses.push_back(stats.trainLoss);
    };
    CrossEntropy loss;
    Adam adam(0.01);
    Matrix t(24, 3);
    for (std::size_t i = 0; i < 24; i++)
        t(i, i % 3) = 1;

    const History h = net.fit({gradient_check::random(24, 4, 3), t}, nullptr, loss, adam, config);
    cr_assert_eq(epochs.size(), 4);
    for (std::size_t e = 0; e < 4; e++) {
        cr_assert_eq(epochs[e], e);
        cr_assert_eq(losses[e], h[e].trainLoss);
    }
}
