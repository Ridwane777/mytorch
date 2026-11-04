#include <criterion/criterion.h>

#include "gradient_check.hpp"

using namespace my_torch;

Test(factory, builds_every_loss)
{
    for (const char *type : {"mse", "mae", "huber", "binary_cross_entropy", "cross_entropy"})
        cr_assert(makeLoss({type, {}})->type() == type, "%s", type);

    const auto loss = makeLoss({"cross_entropy", {{"smoothing", "0.1"}, {"weights", "1,2"}}});
    const CrossEntropy reference(0.1, {1, 2});
    Matrix p(2, 2, 0.5);
    Matrix t(2, 2);
    t(0, 0) = 1;
    t(1, 1) = 1;
    cr_assert_eq(loss->value(p, t), reference.value(p, t));
}

Test(factory, builds_every_optimizer_with_its_rate)
{
    cr_assert_eq(makeOptimizer({"sgd", {{"lr", "0.5"}, {"momentum", "0.9"}, {"nesterov", "1"}}})
                     ->learningRate(),
                 0.5);
    cr_assert_eq(makeOptimizer({"adam", {{"lr", "0.002"}, {"amsgrad", "1"}}})->learningRate(),
                 0.002);
    cr_assert_eq(makeOptimizer({"adamw", {{"weight_decay", "0.1"}}})->type(), "adamw");
    cr_assert_eq(makeOptimizer({"rmsprop", {}})->learningRate(), 0.001);
    cr_assert_eq(makeOptimizer({"adagrad", {}})->learningRate(), 0.01);
}

Test(factory, builds_schedulers_with_warmup)
{
    cr_assert_null(makeScheduler({"none", {}}).get());
    const auto cosine = makeScheduler({"cosine", {{"base", "0.1"}, {"epochs", "10"}}});
    cr_assert_float_eq(cosine->rate(5), CosineAnnealing(0.1, 10).rate(5), 1e-15);
    const auto warm = makeScheduler({"step", {{"base", "0.1"}, {"step", "2"}, {"warmup", "4"}}});
    cr_assert_float_eq(warm->rate(0), 0.1 / 4, 1e-15);
    cr_assert_float_eq(warm->rate(4), 0.1, 1e-15);
    cr_assert(makeScheduler({"exponential", {{"gamma", "0.5"}}})->type() == "exponential");
    cr_assert(makeScheduler({"constant", {{"base", "0.2"}}})->rate(3) == 0.2);
}

Test(factory, rejects_unknown_or_invalid_settings)
{
    cr_assert_throw(makeLoss({"focal", {}}), Error);
    cr_assert_throw(makeOptimizer({"lion", {}}), Error);
    cr_assert_throw(makeScheduler({"cyclic", {}}), Error);
    cr_assert_throw(makeOptimizer({"adam", {{"ir", "0.1"}}}), Error); // typo of lr
    cr_assert_throw(makeOptimizer({"adam", {{"lr", "fast"}}}), Error);
    cr_assert_throw(makeOptimizer({"sgd", {{"nesterov", "2"}}}), Error);
    cr_assert_throw(makeOptimizer({"sgd", {{"lr", "-1"}}}), Error); // constructor check
    cr_assert_throw(makeLoss({"cross_entropy", {{"weights", "1,x"}}}), Error);
    cr_assert_throw(makeScheduler({"cosine", {{"epochs", "-3"}}}), Error);
    cr_assert_throw(makeScheduler({"none", {{"base", "1"}}}), Error);
}
