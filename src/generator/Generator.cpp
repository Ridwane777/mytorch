#include "Generator.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>

#include "my_torch/my_torch.hpp"

namespace generator {

namespace {

constexpr const char *usage =
    "USAGE\n"
    "    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n"
    "\n"
    "DESCRIPTION\n"
    "    config_file_i  Configuration file describing a neural network (docs/configuration.md).\n"
    "    nb_i           Number of neural networks to generate from it (at least 1).\n"
    "\n"
    "    Writes <config name>_1.nn ... <config name>_nb.nn in the current directory, each with\n"
    "    its own initial weights. The same command always writes the same files.\n";

struct Request {
    std::string config;
    std::uint64_t count;
};

std::uint64_t parseCount(const std::string &text)
{
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0)
        throw my_torch::Error("'" + text + "' is not a number of networks (a positive integer)");
    return value;
}

} // namespace

int run(const std::vector<std::string> &args)
{
    if (args.size() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        std::cout << usage;
        return 0;
    }
    if (args.empty() || args.size() % 2 != 0) {
        std::cerr << "my_torch_generator: expected pairs of configuration file and count\n"
                  << usage;
        return EXIT_ERROR;
    }
    std::vector<Request> requests;
    for (std::size_t i = 0; i < args.size(); i += 2)
        requests.push_back({args[i], parseCount(args[i + 1])});
    // Check every configuration before writing anything.
    for (const Request &r : requests)
        my_torch::Network::fromConfig(r.config);
    for (const Request &r : requests) {
        const std::string stem = std::filesystem::path(r.config).stem().string();
        for (std::uint64_t n = 1; n <= r.count; n++)
            my_torch::Network::fromConfig(r.config, n - 1)
                .save(stem + "_" + std::to_string(n) + ".nn");
    }
    return 0;
}

} // namespace generator
