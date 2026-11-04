#include <criterion/criterion.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: one file name per process.
const std::string file = "convolution_network_test_" + std::to_string(getpid());

std::string writeFile(const std::string &name, const std::string &text)
{
    std::ofstream(name) << text;
    return name;
}

// Message of the my_torch::Error thrown by fromConfig on this text, "" if it loads.
std::string configError(const std::string &text)
{
    const std::string path = writeFile(file + ".conf", text);
    std::string message;
    try {
        Network::fromConfig(path);
    } catch (const Error &error) {
        message = error.what();
    }
    std::remove(path.c_str());
    return message;
}

std::string loadError(const std::string &text)
{
    const std::string path = writeFile(file + ".nn", text);
    std::string message;
    try {
        Network::load(path);
    } catch (const Error &error) {
        message = error.what();
    }
    std::remove(path.c_str());
    return message;
}

const char *header = "input raw 129\nlabels 3\nA\nB\nC\n";

} // namespace

// ---- configuration, network file, summary ------------------------------------------------------

namespace {

const std::string cnnConfig =
    std::string(header) +
    "loss cross_entropy\noptimizer adam lr=0.01\ntrain epochs=2 batch=4 seed=3\ninit seed=5\n"
    "layers 8\n"
    "board_planes inputs=129 scalars=64 count=1\n"
    "conv2d channels=3 height=8 width=8 filters=4 kernel=3 stride=1 padding=1 init=he_normal\n"
    "relu\n"
    "max_pool2d channels=4 height=8 width=8 size=2 stride=2\n"
    "avg_pool2d channels=4 height=4 width=4 size=2 stride=2\n"
    "conv2d channels=4 height=2 width=2 filters=3 kernel=2 stride=1 padding=0\n"
    "global_avg_pool channels=3 height=1 width=1\n"
    "softmax\n";

Network cnn()
{
    const std::string path = writeFile(file + ".conf", cnnConfig);
    Network net = Network::fromConfig(path);
    std::remove(path.c_str());
    return net;
}

} // namespace

Test(convolution_config, every_type_is_read_from_a_configuration)
{
    Network net = cnn();
    const std::vector<std::string> types = {"board_planes", "conv2d", "relu", "max_pool2d",
                                            "avg_pool2d", "conv2d", "global_avg_pool", "softmax"};

    cr_assert_eq(net.layers().size(), types.size());
    for (std::size_t i = 0; i < types.size(); i++)
        cr_assert(net.layers()[i]->type() == types[i], "layer %zu: expected %s", i,
                  types[i].c_str());
    const auto *conv = dynamic_cast<const Conv2D *>(net.layers()[1].get());
    cr_assert_not_null(conv);
    cr_assert_eq(conv->weights().rows(), 4);
    cr_assert_eq(conv->weights().cols(), 27);
    cr_assert_eq(conv->outputShape().height, 8);
    // The same configuration and index give the same weights, another index other ones.
    Network again = cnn();
    cr_assert_eq(dynamic_cast<const Conv2D *>(again.layers()[1].get())->weights()(2, 5),
                 conv->weights()(2, 5));
    const Matrix out = net.predict(gradient_check::random(2, 129, 1, 0, 1));
    cr_assert_eq(out.cols(), 3);
    cr_assert_float_eq(out(0, 0) + out(0, 1) + out(0, 2), 1, 1e-12);
}

Test(convolution_config, a_saved_cnn_loads_with_bit_identical_predictions)
{
    Network net = cnn();
    const Matrix input = gradient_check::random(5, 129, 11, 0, 1);
    const Matrix before = net.predict(input);

    net.save(file + ".nn");
    Network loaded = Network::load(file + ".nn");
    const Matrix after = loaded.predict(input);
    std::remove((file + ".nn").c_str());

    cr_assert_eq(before.rows(), after.rows());
    for (std::size_t i = 0; i < before.size(); i++)
        cr_assert_eq(before.data()[i], after.data()[i], "output %zu", i);
    const auto a = net.parameters();
    const auto b = loaded.parameters();
    cr_assert_eq(a.size(), b.size());
    for (std::size_t p = 0; p < a.size(); p++)
        for (std::size_t i = 0; i < a[p].value->size(); i++)
            cr_assert_eq(a[p].value->data()[i], b[p].value->data()[i]);
}

Test(convolution_config, model_summary_follows_the_widths_and_counts_the_parameters)
{
    Network net = cnn();
    std::istringstream lines(modelSummary(net));
    std::string line;
    std::vector<std::vector<std::string>> table;

    std::getline(lines, line); // header
    while (std::getline(lines, line)) {
        std::istringstream words(line);
        table.emplace_back(std::istream_iterator<std::string>(words),
                           std::istream_iterator<std::string>());
    }
    // index, type, output width, parameters; 4·27 + 4 = 112 and 3·16 + 3 = 51
    const std::vector<std::vector<std::string>> expected = {
        {"1", "board_planes", "192", "0"},   {"2", "conv2d", "256", "112"},
        {"3", "relu", "256", "0"},           {"4", "max_pool2d", "64", "0"},
        {"5", "avg_pool2d", "16", "0"},      {"6", "conv2d", "3", "51"},
        {"7", "global_avg_pool", "3", "0"},  {"8", "softmax", "3", "0"},
        {"total", "parameters:", "163"},
    };
    cr_assert_eq(table.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); i++)
        cr_assert(table[i] == expected[i], "row %zu", i);
}

Test(convolution_config, widths_that_do_not_chain_are_rejected)
{
    // A dense layer of 901 outputs followed by a convolution that expects 19 × 8 × 8 = 1216.
    std::string message = configError(
        "input raw 901\nlabels 2\nA\nB\nlayers 3\ndense 901 901\n"
        "conv2d channels=19 height=8 width=8 filters=2 kernel=3 stride=1 padding=1\nsoftmax\n");
    cr_assert(message.find("layer 2 conv2d expects 1216 inputs but receives 901") !=
                  std::string::npos,
              "%s", message.c_str());
    // board_planes on an input of another width.
    message = configError("input raw 900\nlabels 2\nA\nB\nlayers 3\n"
                          "board_planes inputs=901 scalars=768 count=5\ndense 1216 2\nsoftmax\n");
    cr_assert(message.find("layer 1 board_planes expects 901 inputs but receives 900") !=
                  std::string::npos,
              "%s", message.c_str());
    // A convolution after an input that is a dense output of another width.
    message = configError("input raw 10\nlabels 2\nA\nB\nlayers 3\ndense 10 5\n"
                          "max_pool2d channels=1 height=2 width=2 size=2 stride=2\nsoftmax\n");
    cr_assert(message.find("layer 2 max_pool2d expects 4 inputs but receives 5") !=
                  std::string::npos,
              "%s", message.c_str());
}

Test(convolution_config, malformed_lines_are_reported_with_their_line)
{
    const std::string base = "input raw 129\nlabels 2\nA\nB\nlayers 2\n";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"conv2d channels=1 height=8 width=8 filters=2 kernel=3 stride=1\n", "missing padding"},
        {"conv2d channels=1 height=8 width=8 filters=2 kernel=3 stride=1 padding=1 foo=2\n",
         "unexpected 'foo=2'"},
        {"conv2d channels=1 height=8 width=8 filters=2 kernel=9 stride=1 padding=0\n",
         "does not fit"},
        {"conv2d channels=1 height=8 width=8 filters=2 kernel=3 stride=0 padding=0\n",
         "stride must be positive"},
        {"conv2d channels=1 height=8 width=8 filters=0 kernel=3 stride=1 padding=0\n",
         "filters must be positive"},
        {"conv2d channels=1 height=8 width=8 filters=2 kernel=3 stride=1 padding=0 init=bogus\n",
         "unknown init"},
        {"conv2d channels=x height=8 width=8 filters=2 kernel=3 stride=1 padding=0\n",
         "not a non-negative integer"},
        {"max_pool2d channels=1 height=8 width=8 size=0 stride=2\n",
         "window size must be positive"},
        {"avg_pool2d channels=1 height=8 width=8 size=9 stride=2\n", "does not fit"},
        {"global_avg_pool channels=1 height=8\n", "missing width"},
        {"board_planes inputs=901 scalars=768\n", "missing count"},
        {"board_planes inputs=901 scalars=700 count=5\n", "multiple of 64"},
    };
    for (const auto &[line, expected] : cases) {
        const std::string message = configError(base + line + "softmax\n");
        cr_assert(message.find(expected) != std::string::npos, "%s: %s", line.c_str(),
                  message.c_str());
        cr_assert(message.find(":6:") != std::string::npos, "no line number: %s", message.c_str());
    }
}

Test(convolution_config, a_network_file_refuses_the_configuration_only_init_key)
{
    Network net = cnn();
    net.save(file + ".nn");
    std::ifstream in(file + ".nn");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::remove((file + ".nn").c_str());
    const std::string line = "padding=1\n";
    const std::size_t at = text.find(line);

    cr_assert_neq(at, std::string::npos);
    text.replace(at, line.size(), "padding=1 init=he_normal\n");
    // The edited file no longer has its checksum: the unexpected key is reported first.
    const std::string message = loadError(text);
    cr_assert(message.find("unexpected 'init=he_normal'") != std::string::npos, "%s",
              message.c_str());
}
