#include <criterion/criterion.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// A linear regression that plain SGD makes diverge on purpose: inputs of 10 with a rate of 0.1
// multiply the error by about 1 - 2 * 0.1 * 10^2 = -19 at each step, so the weights overflow
// after a few hundred steps.
Dataset divergingData()
{
    Matrix x(4, 1, 10);
    Matrix t(4, 1);
    for (std::size_t i = 0; i < 4; i++)
        t(i, 0) = static_cast<Scalar>(i);
    return {x, t};
}

Network linear()
{
    Network net;
    net.add(std::make_unique<Dense>(1, 1, 3));
    return net;
}

TrainConfig fullBatch(std::size_t epochs)
{
    TrainConfig config;
    config.epochs = epochs;
    config.batchSize = 4;
    config.shuffle = false;
    return config;
}

bool weightsFinite(Network &net)
{
    for (const Parameter &p : net.parameters())
        for (std::size_t i = 0; i < p.value->size(); i++)
            if (!std::isfinite(p.value->data()[i]))
                return false;
    return true;
}

// Records the gradients it receives and updates nothing.
class Spy : public Optimizer {
  public:
    void step(const std::vector<Parameter> &parameters) override
    {
        received.clear();
        for (const Parameter &p : parameters)
            for (std::size_t i = 0; i < p.grad->size(); i++)
                received.push_back(p.grad->data()[i]);
    }
    void setLearningRate(Scalar) override {}
    Scalar learningRate() const override
    {
        return 0;
    }
    std::string type() const override
    {
        return "spy";
    }

    std::vector<Scalar> received;
};

// An identity layer with one parameter whose gradient is NaN: a fault in a known layer.
class Poison : public Layer {
  public:
    Matrix forward(const Matrix &input, bool) override
    {
        return input;
    }
    Matrix backward(const Matrix &gradOutput) override
    {
        _grad(0, 0) = std::numeric_limits<Scalar>::quiet_NaN();
        return gradOutput;
    }
    std::vector<Parameter> parameters() override
    {
        return {{&_value, &_grad}};
    }
    std::string type() const override
    {
        return "poison";
    }

  private:
    Matrix _value{1, 1};
    Matrix _grad{1, 1};
};

std::string fitError(Network &net, const Dataset &data, Optimizer &optimizer,
                     const TrainConfig &config)
{
    MSE loss;
    try {
        net.fit(data, nullptr, loss, optimizer, config);
    } catch (const Error &e) {
        return e.what();
    }
    return "";
}

// Gradients given to the optimizer at the first step of a network with large gradients.
std::vector<Scalar> firstGradients(Scalar clipNorm, Scalar clipValue)
{
    Network net;
    net.add(std::make_unique<Dense>(1, 3, 5));
    net.add(std::make_unique<Dense>(3, 1, 6));
    Spy spy;
    TrainConfig config = fullBatch(1);
    config.clipNorm = clipNorm;
    config.clipValue = clipValue;
    MSE loss;
    net.fit(divergingData(), nullptr, loss, spy, config);
    return spy.received;
}

Scalar norm(const std::vector<Scalar> &v)
{
    Scalar squares = 0;
    for (Scalar x : v)
        squares += x * x;
    return std::sqrt(squares);
}

} // namespace

Test(stability, diverging_training_stops_with_the_epoch)
{
    Network net = linear();
    SGD sgd(0.1);

    const std::string message = fitError(net, divergingData(), sgd, fullBatch(1000));
    cr_assert(message.find("non-finite") != std::string::npos, "got '%s'", message.c_str());
    cr_assert(message.find("at epoch ") != std::string::npos, "got '%s'", message.c_str());
    cr_assert(message.find("clip_norm") != std::string::npos, "got '%s'", message.c_str());
}

Test(stability, clipping_by_norm_keeps_the_diverging_case_finite)
{
    Network net = linear();
    SGD sgd(0.1);
    TrainConfig config = fullBatch(1000);
    config.clipNorm = 1;

    cr_assert_eq(fitError(net, divergingData(), sgd, config), "");
    cr_assert(weightsFinite(net));
}

Test(stability, clipping_by_value_keeps_the_diverging_case_finite)
{
    Network net = linear();
    SGD sgd(0.1);
    TrainConfig config = fullBatch(1000);
    config.clipValue = 1;

    cr_assert_eq(fitError(net, divergingData(), sgd, config), "");
    cr_assert(weightsFinite(net));
}

Test(stability, names_the_layer_with_a_non_finite_gradient)
{
    Network net;
    net.add(std::make_unique<Dense>(1, 1, 3));
    net.add(std::make_unique<Poison>());
    SGD sgd(0.01);

    cr_assert_eq(fitError(net, divergingData(), sgd, fullBatch(1)),
                 "fit: non-finite gradient in layer 2 (poison) at epoch 1, batch 1; lower the "
                 "learning rate or set clip_norm / clip_value");
}

Test(stability, clip_norm_rescales_to_the_threshold_and_keeps_the_direction)
{
    const std::vector<Scalar> raw = firstGradients(0, 0);
    const std::vector<Scalar> clipped = firstGradients(0.5, 0);

    cr_assert_gt(norm(raw), 0.5, "the case must need clipping");
    cr_assert_float_eq(norm(clipped), 0.5, 1e-12);
    for (std::size_t i = 0; i < raw.size(); i++)
        cr_assert_float_eq(clipped[i], raw[i] * 0.5 / norm(raw), 1e-12);
}

Test(stability, clip_norm_leaves_small_gradients_unchanged)
{
    const std::vector<Scalar> raw = firstGradients(0, 0);

    cr_assert(firstGradients(norm(raw) * 2, 0) == raw);
}

Test(stability, clip_value_clamps_each_gradient)
{
    const std::vector<Scalar> raw = firstGradients(0, 0);
    const std::vector<Scalar> clipped = firstGradients(0, 0.1);
    bool someClamped = false;

    for (std::size_t i = 0; i < raw.size(); i++) {
        cr_assert_float_eq(clipped[i], std::clamp(raw[i], -0.1, 0.1), 1e-15);
        someClamped = someClamped || std::abs(raw[i]) > 0.1;
    }
    cr_assert(someClamped, "the case must need clipping");
}

Test(stability, rejects_negative_thresholds)
{
    Network net = linear();
    SGD sgd(0.1);
    TrainConfig config = fullBatch(1);
    config.clipNorm = -1;

    cr_assert(fitError(net, divergingData(), sgd, config).find("clip_norm") != std::string::npos);
}

Test(stability, clipping_is_saved_and_loaded)
{
    const std::string path = "test_stability.nn";
    Network net = linear();
    net.info().inputs = 1;
    net.info().encoding = "raw";
    net.info().labels = {"y"};
    net.info().training.config.clipNorm = 5;
    net.info().training.config.clipValue = 0.25;

    net.save(path);
    const Network loaded = Network::load(path);
    std::remove(path.c_str());
    cr_assert_eq(loaded.info().training.config.clipNorm, 5);
    cr_assert_eq(loaded.info().training.config.clipValue, 0.25);
}

Test(stability, rejects_a_negative_threshold_in_a_file)
{
    const std::string path = "test_stability.conf";
    std::ofstream(path) << "input raw 1\nlabels 1\ny\ntrain clip_norm=-2\nlayers 1\ndense 1 1\n";

    try {
        Network::fromConfig(path);
        cr_assert_fail("a negative clip_norm was accepted");
    } catch (const Error &e) {
        cr_assert(std::string(e.what()).find("clip_norm must be positive") != std::string::npos,
                  "got '%s'", e.what());
    }
    std::remove(path.c_str());
}
