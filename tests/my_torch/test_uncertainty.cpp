#include <criterion/criterion.h>

#include <cmath>
#include <memory>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Dense, optional BatchNorm, ReLU, Dropout(rate), Dense, Softmax, trained a little so that the
// BatchNorm running averages are not their initial values.
Network network(Scalar rate, bool batchNorm = false)
{
    Network net;
    net.add(std::make_unique<Dense>(4, 16, 1));
    if (batchNorm)
        net.add(std::make_unique<BatchNorm>(16));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dropout>(rate, 5));
    net.add(std::make_unique<Dense>(16, 3, 2));
    net.add(std::make_unique<Softmax>());
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

const Matrix input = gradient_check::random(10, 4, 9);

} // namespace

Test(uncertainty, the_same_seed_gives_the_same_result)
{
    Network net = network(0.3);
    const UncertainPrediction a = net.predictMonteCarlo(input, 20, 7);
    const UncertainPrediction b = net.predictMonteCarlo(input, 20, 7);
    const UncertainPrediction c = net.predictMonteCarlo(input, 20, 8);

    bool differs = false;
    for (std::size_t i = 0; i < a.mean.size(); i++) {
        cr_assert_eq(a.mean.data()[i], b.mean.data()[i]);
        differs = differs || a.mean.data()[i] != c.mean.data()[i];
    }
    cr_assert(a.entropy == b.entropy && a.mutualInformation == b.mutualInformation);
    cr_assert(differs, "another seed gave the same passes");
}

Test(uncertainty, leaves_the_network_generators_unchanged)
{
    Network used = network(0.3);
    Network untouched = network(0.3);
    used.predictMonteCarlo(input, 10, 3);

    // The next training forward draws the same dropout masks in both networks.
    const Matrix a = used.forward(input, true);
    const Matrix b = untouched.forward(input, true);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i]);
}

Test(uncertainty, measures_are_consistent)
{
    Network net = network(0.5);
    const UncertainPrediction u = net.predictMonteCarlo(input, 30, 1);
    bool someUncertainty = false;

    for (std::size_t r = 0; r < input.rows(); r++) {
        Scalar sum = 0;
        for (std::size_t c = 0; c < 3; c++)
            sum += u.mean(r, c);
        cr_assert_float_eq(sum, 1, 1e-12);
        cr_assert(u.entropy[r] >= 0 && u.entropy[r] <= std::log(3.0) + 1e-12);
        cr_assert(u.mutualInformation[r] >= 0 && u.mutualInformation[r] <= u.entropy[r] + 1e-12);
        cr_assert(u.disagreement[r] >= 0 && u.disagreement[r] < 1);
        someUncertainty = someUncertainty || u.mutualInformation[r] > 1e-6;
    }
    cr_assert(someUncertainty, "dropout 0.5 gave identical passes");
}

Test(uncertainty, without_dropping_it_is_the_plain_prediction)
{
    // Dropout rate 0: every pass is the prediction, BatchNorm included (prediction mode).
    Network net = network(0, true);
    const UncertainPrediction u = net.predictMonteCarlo(input, 5, 1);
    const Matrix plain = net.predict(input);

    for (std::size_t i = 0; i < plain.size(); i++)
        cr_assert_float_eq(u.mean.data()[i], plain.data()[i], 1e-12);
    for (std::size_t r = 0; r < input.rows(); r++) {
        cr_assert_float_eq(u.mutualInformation[r], 0, 1e-12);
        cr_assert_eq(u.disagreement[r], 0);
    }
}

Test(uncertainty, rejects_no_pass_and_no_dropout)
{
    Network net = network(0.3);
    cr_assert_throw(net.predictMonteCarlo(input, 0), Error);

    Network plain;
    plain.add(std::make_unique<Dense>(4, 3, 1));
    plain.add(std::make_unique<Softmax>());
    cr_assert_throw(plain.predictMonteCarlo(input, 10), Error);
}
