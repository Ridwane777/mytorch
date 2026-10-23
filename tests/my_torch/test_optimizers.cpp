#include <criterion/criterion.h>

#include <cmath>
#include <memory>

#include "my_torch/my_torch.hpp"

using my_torch::Adagrad;
using my_torch::Adam;
using my_torch::AdamW;
using my_torch::Dense;
using my_torch::Matrix;
using my_torch::Optimizer;
using my_torch::Parameter;
using my_torch::RMSProp;
using my_torch::Scalar;
using my_torch::SGD;

namespace {

// Un paramètre scalaire (1 × 1) et son gradient.
struct One {
    Matrix value;
    Matrix grad;

    explicit One(Scalar w, Scalar g = 0) : value(1, 1, w), grad(1, 1, g)
    {
    }

    std::vector<Parameter> params()
    {
        return {{&value, &grad}};
    }
};

// Problème simple : f(w) = Σ c_i·(w_i - a_i)², minimum en a. Les courbures c valent 1 et 10, donc
// la vallée est étroite dans une direction : un test sévère pour un pas fixe.
struct Quadratic {
    Matrix w{1, 2};
    Matrix grad{1, 2};
    Matrix target{1, 2};
    Matrix curvature{1, 2};

    Quadratic()
    {
        w(0, 0) = 4;
        w(0, 1) = -3;
        target(0, 0) = 1;
        target(0, 1) = 2;
        curvature(0, 0) = 1;
        curvature(0, 1) = 10;
    }

    void computeGradient()
    {
        for (std::size_t i = 0; i < 2; i++)
            grad(0, i) = 2 * curvature(0, i) * (w(0, i) - target(0, i));
    }

    // Distance au minimum.
    Scalar error() const
    {
        return std::hypot(w(0, 0) - target(0, 0), w(0, 1) - target(0, 1));
    }

    Scalar run(Optimizer &optimizer, int steps)
    {
        for (int i = 0; i < steps; i++) {
            computeGradient();
            optimizer.step({{&w, &grad}});
        }
        return error();
    }
};

// XOR avec un petit réseau 2-8-1, appris en batch complet : tout le chemin
// forward / coût / backward / optimiseur sur un problème qui n'est pas linéairement séparable.
// Renvoie le nombre d'exemples (sur 4) du bon côté de 0,5.
int learnXor(Optimizer &optimizer, int epochs)
{
    Dense first(2, 8, 1);
    my_torch::Tanh hidden;
    Dense second(8, 1, 2);
    my_torch::Sigmoid output;
    Matrix x(4, 2);
    Matrix y(4, 1);
    const Scalar table[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};

    for (std::size_t i = 0; i < 4; i++) {
        x(i, 0) = table[i][0];
        x(i, 1) = table[i][1];
        y(i, 0) = table[i][2];
    }
    std::vector<Parameter> params = first.parameters();
    for (const Parameter &p : second.parameters())
        params.push_back(p);
    for (int e = 0; e < epochs; e++) {
        for (const Parameter &p : params)
            p.grad->fill(0);
        Matrix p = output.forward(
            second.forward(hidden.forward(first.forward(x, true), true), true), true);
        Matrix g(4, 1);
        for (std::size_t i = 0; i < 4; i++)
            g(i, 0) = 2 * (p(i, 0) - y(i, 0)) / 4;
        first.backward(hidden.backward(second.backward(output.backward(g))));
        optimizer.step(params);
    }
    Matrix p = output.forward(second.forward(hidden.forward(first.forward(x, false), false), false),
                              false);
    int correct = 0;

    for (std::size_t i = 0; i < 4; i++)
        correct += (p(i, 0) > 0.5) == (y(i, 0) > 0.5);
    return correct;
}

} // namespace

Test(sgd, one_step_moves_against_the_gradient)
{
    One w(1, 0.5);
    SGD sgd(0.1);

    sgd.step(w.params());
    cr_assert_float_eq(w.value(0, 0), 0.95, 1e-12);
}

Test(sgd, momentum_accumulates_velocity)
{
    One w(0, 1);
    SGD sgd(0.1, 0.9);

    sgd.step(w.params()); // v = 1,   w = -0.1
    sgd.step(w.params()); // v = 1.9, w = -0.1 - 0.19
    cr_assert_float_eq(w.value(0, 0), -0.29, 1e-12);
}

Test(sgd, nesterov_looks_ahead)
{
    One w(0, 1);
    SGD sgd(0.1, 0.9, true);

    sgd.step(w.params()); // v = 1, pas = g + 0.9·v = 1.9
    cr_assert_float_eq(w.value(0, 0), -0.19, 1e-12);
}

Test(sgd, rejects_bad_hyperparameters)
{
    cr_assert_throw(SGD(0), my_torch::Error);
    cr_assert_throw(SGD(0.1, 1), my_torch::Error);
    cr_assert_throw(SGD(0.1, 0, true), my_torch::Error);
}

Test(adam, first_step_is_about_the_learning_rate)
{
    // Après correction du biais, m̂ = g et v̂ = g² : le pas vaut lr·g / (|g| + ε) ≈ lr, pour
    // n'importe quelle échelle du gradient.
    One small(1, 1e-3);
    One large(1, 1e3);
    Adam a(0.01);
    Adam b(0.01);

    a.step(small.params());
    b.step(large.params());
    cr_assert_float_eq(small.value(0, 0), 1 - 0.01, 1e-6);
    cr_assert_float_eq(large.value(0, 0), 1 - 0.01, 1e-6);
}

Test(adam, matches_a_two_step_hand_computation)
{
    One w(0, 1);
    Adam adam(0.1, 0.9, 0.999, 1e-8);

    adam.step(w.params());
    // Deuxième pas avec un gradient 3 : m = 0.9·0.1 + 0.1·3 = 0.39, v = 0.999·0.001 + 0.001·9.
    w.grad(0, 0) = 3;
    adam.step(w.params());
    const Scalar mHat = 0.39 / (1 - 0.81);
    const Scalar vHat = (0.999 * 0.001 + 0.001 * 9) / (1 - 0.999 * 0.999);

    const Scalar first = -0.1 / (1 + 1e-8); // m̂ = 1, v̂ = 1 au premier pas

    cr_assert_float_eq(w.value(0, 0), first - 0.1 * mHat / (std::sqrt(vHat) + 1e-8), 1e-12);
}

Test(adam, amsgrad_never_lets_the_second_moment_shrink)
{
    One plain(0, 10);
    One ams(0, 10);
    Adam adam(0.1);
    Adam amsgrad(0.1, 0.9, 0.999, 1e-8, true);

    adam.step(plain.params());
    amsgrad.step(ams.params());
    plain.grad(0, 0) = ams.grad(0, 0) = 0.01;
    for (int i = 0; i < 20; i++) {
        adam.step(plain.params());
        amsgrad.step(ams.params());
    }
    // Sans amsgrad, v décroît et le pas regrossit ; avec, il reste plus petit.
    cr_assert_gt(std::abs(plain.value(0, 0)), std::abs(ams.value(0, 0)));
}

Test(adam, adamw_decays_weights_even_with_a_zero_gradient)
{
    One w(1, 0);
    AdamW adamw(0.1, 0.1);
    One untouched(1, 0);
    Adam adam(0.1);

    adamw.step(w.params());
    adam.step(untouched.params());
    cr_assert_float_eq(w.value(0, 0), 1 - 0.1 * 0.1, 1e-12);
    cr_assert_float_eq(untouched.value(0, 0), 1, 1e-12);
    cr_assert_str_eq(adamw.type().c_str(), "adamw");
    cr_assert_str_eq(adam.type().c_str(), "adam");
}

Test(rmsprop, first_step_matches_hand_computation)
{
    One w(0, 2);
    RMSProp rms(0.1, 0.9);

    rms.step(w.params()); // s = 0.1·4 = 0.4, pas = 0.1·2 / (√0.4 + 1e-8)
    cr_assert_float_eq(w.value(0, 0), -0.2 / std::sqrt(0.4), 1e-6);
}

Test(adagrad, step_shrinks_as_squares_accumulate)
{
    One w(0, 1);
    Adagrad adagrad(1.0);

    adagrad.step(w.params()); // s = 1, pas 1
    adagrad.step(w.params()); // s = 2, pas 1/√2
    cr_assert_float_eq(w.value(0, 0), -1 - 1 / std::sqrt(2.0), 1e-6);
}

Test(optimizers, learning_rate_is_readable_and_settable)
{
    SGD sgd(0.1);
    Adam adam;
    RMSProp rms;
    Adagrad ada;
    AdamW adamw;

    for (Optimizer *o : std::initializer_list<Optimizer *>{&sgd, &adam, &rms, &ada, &adamw}) {
        o->setLearningRate(0.5);
        cr_assert_float_eq(o->learningRate(), 0.5, 1e-15);
        cr_assert_throw(o->setLearningRate(0), my_torch::Error);
        cr_assert_throw(o->setLearningRate(-1), my_torch::Error);
    }
}

Test(optimizers, reject_bad_hyperparameters)
{
    cr_assert_throw(Adam(0), my_torch::Error);
    cr_assert_throw(Adam(0.1, 1), my_torch::Error);
    cr_assert_throw(Adam(0.1, 0.9, 1), my_torch::Error);
    cr_assert_throw(AdamW(0.1, -1), my_torch::Error);
    cr_assert_throw(RMSProp(0.1, 1), my_torch::Error);
    cr_assert_throw(Adagrad(0.1, 0), my_torch::Error);
}

Test(optimizers, detect_parameters_that_change_between_steps)
{
    One a(1, 1);
    One b(1, 1);
    SGD sgd(0.1, 0.9);

    sgd.step(a.params());
    cr_assert_throw(sgd.step({{&a.value, &a.grad}, {&b.value, &b.grad}}), my_torch::Error);

    Matrix wide(1, 2);
    SGD other(0.1);
    other.step(a.params());
    cr_assert_throw(other.step({{&wide, &wide}}), my_torch::Error);
    Matrix bad(1, 3);
    SGD third(0.1);
    cr_assert_throw(third.step({{&wide, &bad}}), my_torch::Error);
}

Test(optimizers, converge_on_a_quadratic)
{
    {
        Quadratic q;
        SGD sgd(0.05);
        cr_assert_lt(q.run(sgd, 500), 1e-6);
    }
    {
        Quadratic q;
        SGD sgd(0.02, 0.9);
        cr_assert_lt(q.run(sgd, 500), 1e-6);
    }
    {
        Quadratic q;
        SGD sgd(0.02, 0.9, true);
        cr_assert_lt(q.run(sgd, 500), 1e-6);
    }
    {
        Quadratic q;
        Adam adam(0.1);
        cr_assert_lt(q.run(adam, 500), 0.05);
    }
    {
        Quadratic q;
        AdamW adamw(0.1, 0);
        cr_assert_lt(q.run(adamw, 500), 0.05);
    }
    {
        Quadratic q;
        RMSProp rms(0.02);
        cr_assert_lt(q.run(rms, 500), 0.05);
    }
    {
        Quadratic q;
        Adagrad adagrad(1.0);
        cr_assert_lt(q.run(adagrad, 500), 0.05);
    }
}

Test(optimizers, momentum_beats_plain_sgd_in_a_narrow_valley)
{
    Quadratic plain;
    Quadratic momentum;
    SGD a(0.005);
    SGD b(0.005, 0.9);

    cr_assert_lt(momentum.run(b, 100), plain.run(a, 100));
}

Test(optimizers, learn_xor)
{
    SGD sgd(0.5, 0.9);
    SGD nesterov(0.5, 0.9, true);
    Adam adam(0.05);
    AdamW adamw(0.05, 0.001);
    RMSProp rms(0.02);
    Adagrad adagrad(0.5);

    cr_assert_eq(learnXor(sgd, 1500), 4);
    cr_assert_eq(learnXor(nesterov, 1500), 4);
    cr_assert_eq(learnXor(adam, 1500), 4);
    cr_assert_eq(learnXor(adamw, 1500), 4);
    cr_assert_eq(learnXor(rms, 1500), 4);
    cr_assert_eq(learnXor(adagrad, 1500), 4);
}
