#include <criterion/criterion.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: one file name per process.
const std::string file = "attention_network_test_" + std::to_string(getpid());

void writeText(const std::string &path, const std::string &text)
{
    std::ofstream(path) << text;
}

// 9 inputs: 4 squares of 2 planes, then one scalar. Four tokens of 2 + 1 features, dim 4, a
// pre-norm block (2 heads) and the dense head.
const std::string header = "input test 9\nlabels 3\na\nb\nc\ninit seed=5\n\n";
const std::string layers = "layers 9\n"
                           "to_tokens inputs=9 scalars=8 count=1 tokens=4\n"
                           "token_dense tokens=4 inputs=3 outputs=4\n"
                           "positional tokens=4 dim=4\n"
                           "residual layers=2\n"
                           "token_norm tokens=4 dim=4\n"
                           "self_attention tokens=4 dim=4 heads=2\n"
                           "residual layers=4\n"
                           "token_norm tokens=4 dim=4\n"
                           "token_dense tokens=4 inputs=4 outputs=8\n"
                           "relu\n"
                           "token_dense tokens=4 inputs=8 outputs=4\n"
                           "dense 16 8\n"
                           "relu\n"
                           "dense 8 3\n"
                           "softmax\n";

Network transformer(std::uint64_t index = 0)
{
    const std::string path = file + "_net.conf";
    writeText(path, header + layers);
    Network net = Network::fromConfig(path, index);
    std::remove(path.c_str());
    return net;
}

// Message of the my_torch::Error thrown by fromConfig on this text, "" if it loads.
std::string configError(const std::string &text)
{
    const std::string path = file + "_bad.conf";
    writeText(path, text);
    std::string message;
    try {
        Network::fromConfig(path);
    } catch (const Error &error) {
        message = error.what();
    }
    std::remove(path.c_str());
    return message;
}

std::string loadError(const std::string &path)
{
    try {
        Network::load(path);
    } catch (const Error &error) {
        return error.what();
    }
    return "";
}

std::string readAll(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

void expectSame(const Matrix &a, const Matrix &b)
{
    cr_assert_eq(a.rows(), b.rows());
    cr_assert_eq(a.cols(), b.cols());
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i]);
}

} // namespace

Test(attention_network, configuration_builds_the_transformer)
{
    Network net = transformer();
    cr_assert_eq(net.layers().size(), 9u);
    cr_assert_eq(net.layers()[0]->type(), "to_tokens");
    cr_assert_eq(net.layers()[1]->type(), "token_dense");
    cr_assert_eq(net.layers()[2]->type(), "positional");
    cr_assert_eq(net.layers()[3]->type(), "residual");
    const auto block = net.layers()[3]->children();
    cr_assert_eq(block.size(), 2u);
    cr_assert_eq(block[0]->type(), "token_norm");
    cr_assert_eq(block[1]->type(), "self_attention");
    const Matrix p = net.predict(gradient_check::random(5, 9, 1));
    cr_assert_eq(p.cols(), 3u);
    for (std::size_t r = 0; r < 5; r++)
        cr_assert_float_eq(p(r, 0) + p(r, 1) + p(r, 2), 1, 1e-12);
}

Test(attention_network, same_index_gives_the_same_network_and_each_layer_its_own_weights)
{
    Network a = transformer(0);
    Network b = transformer(0);
    Network c = transformer(1);
    const Matrix x = gradient_check::random(3, 9, 2);
    expectSame(a.predict(x), b.predict(x));
    const Matrix pa = a.predict(x);
    const Matrix pc = c.predict(x);
    bool different = false;
    for (std::size_t i = 0; i < pa.size(); i++)
        different = different || pa.data()[i] != pc.data()[i];
    cr_assert(different);

    // The two token_dense layers of the feed-forward branch draw different weights.
    const auto feedForward = a.layers()[4]->children();
    cr_assert_neq((*feedForward[1]->parameters()[0].value)(0, 0),
                  (*feedForward[3]->parameters()[0].value)(0, 0));
    // The four projections of an attention layer differ from one another.
    auto *attention = dynamic_cast<SelfAttention *>(a.layers()[3]->children()[1]);
    cr_assert_not_null(attention);
    cr_assert_neq(attention->weights(0)(0, 0), attention->weights(1)(0, 0));
}

Test(attention_network, text_and_binary_round_trips_predict_the_same)
{
    Network net = transformer();
    const Matrix x = gradient_check::random(6, 9, 3);
    const Matrix before = net.predict(x);

    for (const bool binary : {true, false}) {
        const std::string path = file + (binary ? ".nn" : ".txt");
        if (binary)
            net.save(path);
        else
            net.saveText(path);
        Network loaded = Network::load(path);
        expectSame(loaded.predict(x), before);
        cr_assert_eq(loaded.layers().size(), 9u);
        cr_assert_eq(loaded.layers()[4]->children()[1]->type(), "token_dense");
        // Saved again, the loaded network gives the same bytes.
        const std::string again = path + ".again";
        if (binary)
            loaded.save(again);
        else
            loaded.saveText(again);
        cr_assert_eq(readAll(again), readAll(path));
        std::remove(again.c_str());
        std::remove(path.c_str());
    }
}

Test(attention_network, binary_file_uses_the_new_keyword_codes)
{
    Network net = transformer();
    const std::string path = file + "_words.nn";
    net.save(path);
    const std::string bytes = readAll(path);
    // The names of the layers are keywords, not strings: the text of a layer name never appears.
    for (const char *name : {"to_tokens", "token_dense", "self_attention", "token_norm"})
        cr_assert_eq(bytes.find(name), std::string::npos, "%s is stored as a string", name);
    std::remove(path.c_str());
}

Test(attention_network, an_old_binary_file_still_loads)
{
    // The keyword table is append only: the CNN snapshot written before the token layers keeps
    // its codes.
    const std::string path = "bonus/networks/my_torch_network_cnn_epoch10.nn";
    if (!std::ifstream(path))
        return;
    Network net = Network::load(path);
    cr_assert_eq(net.layers()[0]->type(), "to_planes");
    const Matrix p = net.predict(Matrix(2, net.info().inputs));
    cr_assert_eq(p.cols(), net.info().labels.size());
    cr_assert_float_eq(p(0, 0) + p(0, 1) + p(0, 2) + p(0, 3) + p(0, 4), 1, 1e-12);
}

Test(attention_network, summary_lists_the_layers_and_their_widths)
{
    Network net = transformer();
    const std::string summary = modelSummary(net);
    cr_assert_neq(summary.find("to_tokens"), std::string::npos);
    cr_assert_neq(summary.find("residual"), std::string::npos); // the blocks are one layer each
    cr_assert_neq(summary.find("positional"), std::string::npos);
}

Test(attention_network, configuration_errors_name_the_layer)
{
    const auto with = [](const std::string &replaced, const std::string &by) {
        std::string text = header + layers;
        const std::size_t at = text.find(replaced);
        text.replace(at, replaced.size(), by);
        return text;
    };
    // dim not divisible by the heads
    cr_assert_neq(configError(with("heads=2", "heads=3")).find("multiple of heads"),
                  std::string::npos);
    // a token layer that does not receive tokens·dim columns
    cr_assert_neq(
        configError(with("token_dense tokens=4 inputs=3", "token_dense tokens=4 inputs=2"))
            .find("expects"),
        std::string::npos);
    // a residual block that changes the width
    cr_assert_neq(configError(with("token_dense tokens=4 inputs=8 outputs=4",
                                   "token_dense tokens=4 inputs=8 outputs=3"))
                      .find("residual"),
                  std::string::npos);
    // a bad to_tokens layout
    cr_assert_neq(configError(with("scalars=8", "scalars=7")).find("to_tokens"), std::string::npos);
    // a missing and an unknown setting
    cr_assert_neq(configError(with(" heads=2", "")).find("missing heads"), std::string::npos);
    cr_assert_neq(configError(with("heads=2", "heads=2 mask=1")).find("unexpected"),
                  std::string::npos);
    cr_assert_neq(configError(with("positional tokens=4 dim=4", "positional tokens=4 dim=0"))
                      .find("positional"),
                  std::string::npos);
}

Test(attention_network, a_truncated_network_file_is_refused)
{
    Network net = transformer();
    const std::string path = file + "_cut.txt";
    net.saveText(path);
    std::string text = readAll(path);
    // Drop the first line of weights of the attention layer, and the checksum with it.
    const std::size_t layer = text.find("self_attention");
    const std::size_t weights = text.find('\n', layer) + 1;
    text.erase(weights, text.find('\n', weights) + 1 - weights);
    text.erase(text.find("end checksum="));
    text += "end\n";
    writeText(path, text);
    cr_assert_neq(loadError(path), "");
    std::remove(path.c_str());
}

Test(attention_network, input_gradient_goes_through_the_attention_layers)
{
    // Saliency calls backward after a prediction forward (ADR-0035).
    Network net = transformer();
    const Matrix x = gradient_check::random(1, 9, 4);
    const Matrix saliency = net.inputGradient(x, 1);
    cr_assert_eq(saliency.cols(), 9u);
    for (std::size_t c = 0; c < 9; c++) {
        Matrix plus = x;
        Matrix minus = x;
        plus(0, c) += 1e-6;
        minus(0, c) -= 1e-6;
        const Scalar numeric = (net.predict(plus)(0, 1) - net.predict(minus)(0, 1)) / 2e-6;
        cr_assert_float_eq(saliency(0, c), numeric, 1e-6, "column %zu", c);
    }
}

Test(attention_network, tiny_network_overfits_a_few_examples)
{
    Network net = transformer();
    // Twelve random positions with arbitrary classes: a transformer of this size memorizes them.
    const Matrix inputs = gradient_check::random(12, 9, 11, 0, 1);
    Matrix targets(12, 3);
    for (std::size_t r = 0; r < 12; r++)
        targets(r, (r * 7 + 1) % 3) = 1;
    CrossEntropy loss;
    Adam adam(0.01);
    TrainConfig config;
    config.epochs = 400;
    config.batchSize = 12;
    config.shuffle = false;

    const History history = net.fit({inputs, targets}, nullptr, loss, adam, config);
    cr_assert_lt(history.back().trainLoss, history.front().trainLoss);
    cr_assert_lt(history.back().trainLoss, 0.1);
    const Matrix p = net.predict(inputs);
    for (std::size_t r = 0; r < 12; r++) {
        std::size_t best = 0;
        for (std::size_t c = 1; c < 3; c++)
            if (p(r, c) > p(r, best))
                best = c;
        cr_assert_eq(best, (r * 7 + 1) % 3, "example %zu", r);
    }
}
