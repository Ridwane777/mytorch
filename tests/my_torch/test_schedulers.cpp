#include <criterion/criterion.h>

#include <cmath>
#include <memory>

#include "my_torch/my_torch.hpp"

using my_torch::ConstantRate;
using my_torch::CosineAnnealing;
using my_torch::ExponentialDecay;
using my_torch::StepDecay;
using my_torch::Warmup;

Test(schedulers, constant_never_changes)
{
    ConstantRate constant(0.1);

    cr_assert_float_eq(constant.rate(0), 0.1, 1e-15);
    cr_assert_float_eq(constant.rate(1000), 0.1, 1e-15);
}

Test(schedulers, step_decay_drops_every_step_size_epochs)
{
    StepDecay step(1.0, 10, 0.5);

    cr_assert_float_eq(step.rate(0), 1.0, 1e-15);
    cr_assert_float_eq(step.rate(9), 1.0, 1e-15);
    cr_assert_float_eq(step.rate(10), 0.5, 1e-15);
    cr_assert_float_eq(step.rate(25), 0.25, 1e-15);
}

Test(schedulers, exponential_decay_multiplies_each_epoch)
{
    ExponentialDecay exponential(2.0, 0.9);

    cr_assert_float_eq(exponential.rate(0), 2.0, 1e-15);
    cr_assert_float_eq(exponential.rate(3), 2.0 * 0.9 * 0.9 * 0.9, 1e-12);
}

Test(schedulers, cosine_follows_a_half_cosine)
{
    CosineAnnealing cosine(1.0, 100, 0.1);

    cr_assert_float_eq(cosine.rate(0), 1.0, 1e-12);
    cr_assert_float_eq(cosine.rate(50), 0.55, 1e-12);
    cr_assert_float_eq(cosine.rate(100), 0.1, 1e-12);
    cr_assert_float_eq(cosine.rate(500), 0.1, 1e-12);
    for (std::size_t e = 1; e <= 100; e++)
        cr_assert_leq(cosine.rate(e), cosine.rate(e - 1));
}

Test(schedulers, warmup_ramps_up_then_hands_over)
{
    Warmup warmup(std::make_unique<CosineAnnealing>(1.0, 100), 4);

    cr_assert_float_eq(warmup.rate(0), 0.25, 1e-12);
    cr_assert_float_eq(warmup.rate(1), 0.5, 1e-12);
    cr_assert_float_eq(warmup.rate(3), 1.0, 1e-12);
    cr_assert_float_eq(warmup.rate(4), 1.0, 1e-12);  // époque 0 du cosinus
    cr_assert_float_eq(warmup.rate(54), 0.5, 1e-12); // époque 50 du cosinus
    cr_assert_str_eq(warmup.type().c_str(), "warmup_cosine");
}

Test(schedulers, rates_drive_an_optimizer)
{
    my_torch::SGD sgd(1.0);
    StepDecay step(1.0, 1, 0.5);

    for (std::size_t epoch = 0; epoch < 3; epoch++)
        sgd.setLearningRate(step.rate(epoch));
    cr_assert_float_eq(sgd.learningRate(), 0.25, 1e-15);
}

Test(schedulers, reject_bad_parameters)
{
    cr_assert_throw(ConstantRate(0), my_torch::Error);
    cr_assert_throw(StepDecay(1, 0), my_torch::Error);
    cr_assert_throw(StepDecay(1, 5, 0), my_torch::Error);
    cr_assert_throw(ExponentialDecay(1, -1), my_torch::Error);
    cr_assert_throw(CosineAnnealing(1, 0), my_torch::Error);
    cr_assert_throw(CosineAnnealing(1, 10, 2), my_torch::Error);
    cr_assert_throw(Warmup(nullptr, 3), my_torch::Error);
    cr_assert_throw(Warmup(std::make_unique<ConstantRate>(1), 0), my_torch::Error);
}
