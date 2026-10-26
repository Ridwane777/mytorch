#include <criterion/criterion.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../src/generator/Generator.hpp"
#include "my_torch/my_torch.hpp"

namespace fs = std::filesystem;

namespace {

// Criterion runs tests in parallel processes, and the generator writes in the current directory:
// every name carries the process id.
std::string name(const std::string &base)
{
    return "gen_" + std::to_string(getpid()) + "_" + base;
}

const std::string header = "input raw 3\nlabels 2\nyes\nno\n";
const std::string body = "layers 3\ndense 3 4\ntanh\ndense 4 2\n";

std::string writeConfig(const std::string &base, const std::string &text)
{
    const std::string path = name(base) + ".conf";
    std::ofstream(path) << text;
    return path;
}

// Same as main: an exception becomes exit code 84.
int exitCode(const std::vector<std::string> &args)
{
    try {
        return generator::run(args);
    } catch (const std::exception &) {
        return generator::EXIT_ERROR;
    }
}

std::string output(const std::string &base, int n)
{
    return name(base) + "_" + std::to_string(n) + ".nn";
}

std::string readText(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

void cleanup(const std::string &base)
{
    std::remove((name(base) + ".conf").c_str());
    for (int n = 1; n <= 5; n++)
        std::remove(output(base, n).c_str());
}

std::string errorOf(const std::string &path)
{
    try {
        my_torch::Network::fromConfig(path);
    } catch (const my_torch::Error &e) {
        return e.what();
    }
    return "";
}

} // namespace

Test(generator, follows_the_subject_example)
{
    const std::string conf = writeConfig("basic_network", header + body);

    cr_assert_eq(exitCode({conf, "3"}), 0);
    for (int n = 1; n <= 3; n++)
        cr_assert(fs::exists(output("basic_network", n)), "missing %d", n);
    cr_assert(!fs::exists(output("basic_network", 4)));
    cleanup("basic_network");
}

Test(generator, written_networks_load_with_the_configuration)
{
    const std::string conf = writeConfig(
        "full", header +
                    "loss cross_entropy smoothing=0.1\noptimizer sgd lr=0.5 momentum=0.9\n"
                    "scheduler step base=0.5 step=3\ntrain epochs=4 batch=2 seed=9\n"
                    "layers 5\ndense 3 4 l2=0.01\nrelu\ndropout rate=0.2\ndense 4 2\nsoftmax\n");

    cr_assert_eq(exitCode({conf, "1"}), 0);
    const my_torch::Network net = my_torch::Network::load(output("full", 1));
    cleanup("full");
    const my_torch::ModelInfo &info = net.info();
    cr_assert(info.encoding == "raw" && info.inputs == 3);
    cr_assert(info.labels == std::vector<std::string>({"yes", "no"}));
    cr_assert(info.training.optimizer.type == "sgd");
    cr_assert(info.training.scheduler.type == "step");
    cr_assert_eq(info.training.config.epochs, 4);
    cr_assert_eq(net.layers().size(), 5);
    cr_assert(net.layers()[2]->type() == "dropout");
}

Test(generator, networks_differ_but_the_command_is_reproducible)
{
    const std::string conf = writeConfig("repro", header + body);

    cr_assert_eq(exitCode({conf, "2"}), 0);
    const std::string first = readText(output("repro", 1));
    const std::string second = readText(output("repro", 2));
    cr_assert(first != second);
    cr_assert_eq(exitCode({conf, "2"}), 0);
    cr_assert(readText(output("repro", 1)) == first);
    cr_assert(readText(output("repro", 2)) == second);
    cleanup("repro");
}

Test(generator, init_seed_changes_the_weights)
{
    const std::string a = writeConfig("seed_a", header + "init seed=1\n" + body);
    const std::string b = writeConfig("seed_b", header + "init seed=2\n" + body);

    cr_assert_eq(exitCode({a, "1", b, "1"}), 0);
    cr_assert(readText(output("seed_a", 1)) != readText(output("seed_b", 1)));
    cleanup("seed_a");
    cleanup("seed_b");
}

Test(generator, several_configurations_in_one_command)
{
    const std::string a = writeConfig("many_a", header + body);
    const std::string b = writeConfig("many_b", header + body);

    cr_assert_eq(exitCode({a, "2", b, "1"}), 0);
    cr_assert(fs::exists(output("many_a", 2)) && fs::exists(output("many_b", 1)));
    cr_assert(!fs::exists(output("many_b", 2)));
    cleanup("many_a");
    cleanup("many_b");
}

Test(generator, bad_arguments_exit_84_and_write_nothing)
{
    const std::string good = writeConfig("args_good", header + body);
    const std::string bad = writeConfig("args_bad", header + "layers 1\nconv2d\n");

    cr_assert_eq(exitCode({}), 84);
    cr_assert_eq(exitCode({good}), 84);
    for (const char *count : {"0", "-2", "abc", "3x"})
        cr_assert_eq(exitCode({good, count}), 84, "count %s", count);
    cr_assert_eq(exitCode({"missing.conf", "1"}), 84);
    // The second configuration is wrong: the first one must not be written either.
    cr_assert_eq(exitCode({good, "2", bad, "1"}), 84);
    cr_assert(!fs::exists(output("args_good", 1)));
    cleanup("args_good");
    cleanup("args_bad");
}

Test(generator, help_exits_0)
{
    cr_assert_eq(exitCode({"--help"}), 0);
    cr_assert_eq(exitCode({"-h"}), 0);
}

Test(generator, configuration_comments_defaults_and_order)
{
    // Comments, blank lines, sections in any order, loss and optimizer left to their defaults.
    const std::string conf = writeConfig(
        "order", "# a comment\n\n" + header + "  # indented comment\ntrain epochs=3\n\n" +
                     "scheduler none\n" + body);
    const my_torch::Network net = my_torch::Network::fromConfig(conf);
    cleanup("order");

    cr_assert(net.info().training.loss.type == "cross_entropy");
    cr_assert(net.info().training.optimizer.type == "adam");
    cr_assert_eq(net.info().training.config.epochs, 3);
}

Test(generator, dropout_seed_is_derived_per_network)
{
    const std::string conf = writeConfig(
        "drop", header + "layers 4\ndense 3 2\ndropout rate=0.5\ndense 2 2\nsoftmax\n");
    const auto seedOf = [&](std::uint64_t index) {
        const my_torch::Network net = my_torch::Network::fromConfig(conf, index);
        return dynamic_cast<const my_torch::Dropout *>(net.layers()[1].get())->seed();
    };

    cr_assert_neq(seedOf(0), seedOf(1));
    cr_assert_eq(seedOf(1), seedOf(1));
    cleanup("drop");
}

Test(generator, configuration_errors_name_the_line)
{
    struct Case {
        std::string text;
        std::string what;
    };
    const std::vector<Case> cases = {
        {header + "speed fast\n" + body, ":5: unknown keyword 'speed'"},
        {header + "train epochs=1\ntrain epochs=2\n" + body, ":6: 'train' given twice"},
        {header + "init depth=3\n" + body, "init: unknown setting 'depth'"},
        {header + "layers 1\ndense 3 2 init=xavier\n", "unknown init 'xavier'"},
        {header + "layers 2\ndense 3 2\n0.5 0.5\n", "unknown layer type '0.5'"},
        {header + "layers 1\ndense 3 2\nrelu\n", "unexpected data after the layers"},
        {header + "optimizer lion\n" + body, "unknown optimizer 'lion'"},
        {header + "layers 1\ndense 4 2\n", "expects 4 inputs but receives 3"},
        {"input raw 3\nlabels 1\nyes\n" + body, "2 outputs for 1 labels"},
    };
    for (std::size_t i = 0; i < cases.size(); i++) {
        const std::string conf = writeConfig("err" + std::to_string(i), cases[i].text);
        const std::string message = errorOf(conf);
        cleanup("err" + std::to_string(i));
        cr_assert(message.find(cases[i].what) != std::string::npos, "%s -> %s",
                  cases[i].what.c_str(), message.c_str());
    }
}

Test(generator, every_init_name_is_accepted_and_changes_the_weights)
{
    const std::vector<std::string> names = {"uniform",        "lecun_uniform", "lecun_normal",
                                            "xavier_uniform", "xavier_normal", "he_uniform",
                                            "he_normal",      "orthogonal"};
    std::vector<std::string> texts;

    for (std::size_t i = 0; i < names.size(); i++) {
        const std::string base = "init" + std::to_string(i);
        const std::string conf =
            writeConfig(base, header + "layers 2\ndense 3 4 init=" + names[i] + "\ndense 4 2\n");

        cr_assert_eq(exitCode({conf, "1"}), 0, "%s", names[i].c_str());
        texts.push_back(readText(output(base, 1)));
        cleanup(base);
    }
    for (std::size_t i = 0; i < texts.size(); i++)
        for (std::size_t j = i + 1; j < texts.size(); j++)
            cr_assert(texts[i] != texts[j], "%s and %s give the same network", names[i].c_str(),
                      names[j].c_str());
}

Test(generator, no_init_key_is_init_uniform)
{
    const std::string plain = writeConfig("noinit", header + "layers 2\ndense 3 4\ndense 4 2\n");
    const std::string named =
        writeConfig("uniform", header + "layers 2\ndense 3 4 init=uniform\ndense 4 2\n");

    cr_assert_eq(exitCode({plain, "1", named, "1"}), 0);
    // The initialisation is not stored in the file: the same seeds give the same bytes.
    cr_assert(readText(output("noinit", 1)) == readText(output("uniform", 1)));
    cleanup("noinit");
    cleanup("uniform");
}

Test(generator, he_normal_layer_has_the_he_variance)
{
    const std::string conf = writeConfig(
        "he", "input raw 400\nlabels 2\nyes\nno\nlayers 2\ndense 400 300 init=he_normal\n"
              "dense 300 2\n");
    const my_torch::Network net = my_torch::Network::fromConfig(conf);
    const auto &dense = dynamic_cast<const my_torch::Dense &>(*net.layers()[0]);
    double square = 0;

    cleanup("he");
    for (std::size_t i = 0; i < dense.weights().size(); i++)
        square += dense.weights().data()[i] * dense.weights().data()[i];
    cr_assert_lt(std::abs(square / dense.weights().size() * 400 / 2 - 1), 0.05);
}

Test(generator, unknown_init_exits_84_with_the_line_and_the_names)
{
    const std::string conf =
        writeConfig("badinit", header + "layers 2\ndense 3 4 init=xavier\ndense 4 2\n");
    const std::string message = errorOf(conf);

    cr_assert(message.find("badinit.conf:6:") != std::string::npos, "%s", message.c_str());
    cr_assert(message.find("unknown init 'xavier'") != std::string::npos, "%s", message.c_str());
    for (const char *known : {"uniform", "lecun_uniform", "lecun_normal", "xavier_uniform",
                              "xavier_normal", "he_uniform", "he_normal", "orthogonal"})
        cr_assert(message.find(known) != std::string::npos, "%s missing: %s", known,
                  message.c_str());
    cr_assert_eq(exitCode({conf, "1"}), generator::EXIT_ERROR);
    cr_assert(!fs::exists(output("badinit", 1)));
    cleanup("badinit");
}

Test(generator, example_configuration_of_the_repository_is_valid)
{
    const my_torch::Network net = my_torch::Network::fromConfig("configs/basic_network.conf");

    cr_assert_eq(net.info().inputs, 768);
    cr_assert_eq(net.info().labels.size(), 5);
}
