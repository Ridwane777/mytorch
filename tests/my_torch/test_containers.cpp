#include <criterion/criterion.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: one file name per process.
const std::string file = "containers_test_" + std::to_string(getpid());

LayerSequence block(std::size_t width, std::uint32_t seed)
{
    LayerSequence layers;
    layers.push_back(std::make_unique<Dense>(width, width, seed));
    layers.push_back(std::make_unique<Tanh>());
    layers.push_back(std::make_unique<Dense>(width, width, seed + 1));
    return layers;
}

std::unique_ptr<Concat> twoBranches(std::size_t width)
{
    std::vector<LayerSequence> branches;
    LayerSequence a;
    a.push_back(std::make_unique<Dense>(width, 2, 5));
    branches.push_back(std::move(a));
    LayerSequence b;
    b.push_back(std::make_unique<Dense>(width, 3, 6));
    b.push_back(std::make_unique<Sigmoid>());
    branches.push_back(std::move(b));
    return std::make_unique<Concat>(std::move(branches));
}

// 4 inputs → residual(4) → concat(2 + 3) → dense 5→3 → softmax, labels a, b, c.
Network network()
{
    Network net;
    net.add(std::make_unique<Residual>(block(4, 1)));
    net.add(twoBranches(4));
    net.add(std::make_unique<Dense>(5, 3, 9));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "test";
    net.info().inputs = 4;
    net.info().labels = {"a", "b", "c"};
    return net;
}

void writeText(const std::string &path, const std::string &text)
{
    std::ofstream(path) << text;
}

} // namespace

Test(containers, residual_adds_the_block_to_its_input)
{
    LayerSequence inner;
    inner.push_back(std::make_unique<Dense>(3, 3, 4));
    Dense reference(3, 3, 4);
    Residual residual(std::move(inner));
    const Matrix x = gradient_check::random(5, 3, 1);
    const Matrix out = residual.forward(x, false);
    const Matrix f = reference.forward(x, false);
    for (std::size_t i = 0; i < out.size(); i++)
        cr_assert_float_eq(out.data()[i], f.data()[i] + x.data()[i], 1e-15);
}

Test(containers, residual_input_gradient)
{
    Residual residual(block(4, 1));
    cr_assert_lt(gradient_check::inputError(residual, gradient_check::random(3, 4, 2), 4),
                 gradient_check::tolerance);
}

Test(containers, concat_puts_the_branches_side_by_side)
{
    const auto concat = twoBranches(4);
    Dense a(4, 2, 5);
    Dense b(4, 3, 6);
    Sigmoid sigmoid;
    const Matrix x = gradient_check::random(2, 4, 3);
    const Matrix out = concat->forward(x, false);
    const Matrix pa = a.forward(x, false);
    const Matrix pb = sigmoid.forward(b.forward(x, false), false);
    cr_assert_eq(out.cols(), 5u);
    for (std::size_t r = 0; r < 2; r++) {
        for (std::size_t c = 0; c < 2; c++)
            cr_assert_float_eq(out(r, c), pa(r, c), 1e-15);
        for (std::size_t c = 0; c < 3; c++)
            cr_assert_float_eq(out(r, 2 + c), pb(r, c), 1e-15);
    }
    cr_assert_eq(concat->outputWidth(4), 5u);
}

Test(containers, concat_input_gradient)
{
    const auto concat = twoBranches(4);
    cr_assert_lt(gradient_check::inputError(*concat, gradient_check::random(3, 4, 4), 5),
                 gradient_check::tolerance);
}

Test(containers, parameter_gradients_through_nested_containers)
{
    // A residual block inside a branch, the branches after another residual block.
    std::vector<LayerSequence> branches;
    LayerSequence deep;
    deep.push_back(std::make_unique<Residual>(block(4, 11)));
    deep.push_back(std::make_unique<Dense>(4, 2, 13));
    branches.push_back(std::move(deep));
    LayerSequence shallow;
    shallow.push_back(std::make_unique<Dense>(4, 2, 14));
    branches.push_back(std::move(shallow));

    gradient_check::Layers layers;
    layers.push_back(std::make_unique<Residual>(block(4, 1)));
    layers.push_back(std::make_unique<Concat>(std::move(branches)));
    layers.push_back(std::make_unique<Dense>(4, 3, 15));
    layers.push_back(std::make_unique<Softmax>());
    cr_assert_lt(gradient_check::networkError(layers, CrossEntropy(),
                                              gradient_check::random(6, 4, 5),
                                              Matrix(6, 3, 1.0 / 3)),
                 gradient_check::tolerance);
}

Test(containers, parameters_and_children_cover_the_inner_layers)
{
    Residual residual(block(4, 1));
    cr_assert_eq(residual.parameters().size(), 4u); // two dense layers: weights and biases
    cr_assert_eq(residual.children().size(), 3u);
    Network net = network();
    cr_assert_eq(net.parameters().size(), 10u);
}

Test(containers, save_and_load_text_and_binary)
{
    Network net = network();
    const Matrix x = gradient_check::random(4, 4, 6);
    const Matrix before = net.predict(x);
    for (const bool binary : {true, false}) {
        const std::string path = file + (binary ? ".nn" : ".txt");
        if (binary)
            net.save(path);
        else
            net.saveText(path);
        Network loaded = Network::load(path);
        const Matrix after = loaded.predict(x);
        for (std::size_t i = 0; i < after.size(); i++)
            cr_assert_eq(after.data()[i], before.data()[i]);
        cr_assert_eq(loaded.layers()[0]->type(), "residual");
        cr_assert_eq(loaded.layers()[1]->type(), "concat");
        std::remove(path.c_str());
    }
}

Test(containers, configuration_with_containers)
{
    const std::string path = file + ".conf";
    writeText(path, "input test 4\nlabels 2\nno\nyes\ninit seed=3\n\nlayers 4\n"
                    "residual layers=3\ndense 4 4\nrelu\ndense 4 4\n"
                    "concat branches=2\nbranch layers=1\ndense 4 2\nbranch layers=2\ndense 4 2\n"
                    "tanh\n"
                    "dense 4 2\nsoftmax\n");
    Network a = Network::fromConfig(path);
    Network b = Network::fromConfig(path);
    const Matrix x = gradient_check::random(3, 4, 7);
    const Matrix pa = a.predict(x);
    const Matrix pb = b.predict(x);
    for (std::size_t i = 0; i < pa.size(); i++)
        cr_assert_eq(pa.data()[i], pb.data()[i]);
    // Each nested dense layer draws its own weights: the two dense 4 4 of the block differ.
    const auto blockLayers = a.layers()[0]->children();
    const Matrix &w1 = *blockLayers[0]->parameters()[0].value;
    const Matrix &w2 = *blockLayers[2]->parameters()[0].value;
    cr_assert_neq(w1(0, 0), w2(0, 0));
    std::remove(path.c_str());
}

Test(containers, widths_are_checked)
{
    LayerSequence narrowing;
    narrowing.push_back(std::make_unique<Dense>(4, 3, 1));
    Residual residual(std::move(narrowing));
    cr_assert_throw(residual.outputWidth(4), Error);
    cr_assert_throw(residual.forward(Matrix(2, 4), false), Error);
    cr_assert_throw(Residual(LayerSequence{}), Error);

    std::vector<LayerSequence> one;
    one.push_back(block(4, 1));
    cr_assert_throw(Concat(std::move(one)), Error);

    const std::string path = file + "_bad.conf";
    writeText(path, "input test 4\nlabels 2\nno\nyes\n\nlayers 3\n"
                    "residual layers=1\ndense 4 3\ndense 3 2\nsoftmax\n");
    cr_assert_throw(Network::fromConfig(path), Error);
    writeText(path,
              "input test 4\nlabels 2\nno\nyes\n\nlayers 3\n"
              "concat branches=2\nbranch layers=1\ndense 4 1\ndense 4 1\ndense 2 2\nsoftmax\n");
    cr_assert_throw(Network::fromConfig(path), Error); // second branch record missing
    std::remove(path.c_str());
}

Test(containers, monte_carlo_dropout_inside_a_block)
{
    LayerSequence inner;
    inner.push_back(std::make_unique<Dense>(4, 4, 1));
    inner.push_back(std::make_unique<Dropout>(0.5, 3));
    Network net;
    net.add(std::make_unique<Residual>(std::move(inner)));
    net.add(std::make_unique<Dense>(4, 3, 2));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "test";
    net.info().inputs = 4;
    net.info().labels = {"a", "b", "c"};
    const Matrix x = gradient_check::random(5, 4, 8);
    const UncertainPrediction p = net.predictMonteCarlo(x, 30, 1);
    const UncertainPrediction q = net.predictMonteCarlo(x, 30, 1);
    for (std::size_t r = 0; r < 5; r++) {
        cr_assert_eq(p.entropy[r], q.entropy[r]);
        cr_assert_gt(p.mutualInformation[r], 0); // the passes differ: the dropout was active
    }
}

Test(containers, a_residual_network_trains)
{
    // 64 points, class = sign of x0 * x1.
    const Matrix x = gradient_check::random(64, 2, 9);
    Dataset data{Matrix(64, 4), Matrix(64, 2)};
    for (std::size_t r = 0; r < 64; r++) {
        data.inputs(r, 0) = x(r, 0);
        data.inputs(r, 1) = x(r, 1);
        data.targets(r, x(r, 0) * x(r, 1) > 0 ? 1 : 0) = 1;
    }
    Network net;
    net.add(std::make_unique<Residual>(block(4, 1)));
    net.add(std::make_unique<Residual>(block(4, 3)));
    net.add(std::make_unique<Dense>(4, 2, 5));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "test";
    net.info().inputs = 4;
    net.info().labels = {"no", "yes"};
    CrossEntropy loss;
    Adam adam(0.01);
    TrainConfig config;
    config.epochs = 200;
    config.batchSize = 16;
    const History history = net.fit(data, nullptr, loss, adam, config);
    cr_assert_lt(history.back().trainLoss, history.front().trainLoss / 2);
    cr_assert_gt(history.back().trainAccuracy, 0.9);
}
