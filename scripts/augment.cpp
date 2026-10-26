// Data augmentation by board symmetry (#38, ADR-0019): reads a labelled chess file, writes it
// back with the mirror and/or the colour-swapped flip of every position.
//
//   augment [--mirror | --flip | --all] IN OUT      (default --all)
//
// Output: "FEN label" per line, in the order original, mirror, flip, mirror+flip. Standalone tool:
// my_torch_analyzer and the network file are untouched. Build (also done by the scripts that use
// it; my_torch/Error.hpp is header-only, so no library source is needed):
//   g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/augment.cpp
//       src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp -o augment
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Fen.hpp"
#include "Symmetry.hpp"
#include "my_torch/Error.hpp"

namespace {

constexpr int EXIT_ERROR = 84;

constexpr const char *usage = "usage: augment [--mirror | --flip | --all] IN OUT\n";

struct Options {
    analyzer::Symmetries symmetries = analyzer::Symmetries::All;
    std::string input;
    std::string output;
};

Options parseArguments(const std::vector<std::string> &args)
{
    Options options;
    std::vector<std::string> files;
    bool chosen = false;

    for (const std::string &arg : args) {
        if (arg == "--mirror" || arg == "--flip" || arg == "--all") {
            if (chosen)
                throw my_torch::Error("only one of --mirror, --flip and --all is allowed");
            chosen = true;
            options.symmetries = arg == "--mirror" ? analyzer::Symmetries::Mirror
                                 : arg == "--flip" ? analyzer::Symmetries::Flip
                                                   : analyzer::Symmetries::All;
        } else if (arg.size() > 1 && arg[0] == '-') {
            throw my_torch::Error("unknown option '" + arg + "'");
        } else {
            files.push_back(arg);
        }
    }
    if (files.size() != 2)
        throw my_torch::Error("expected an input file and an output file");
    options.input = files[0];
    options.output = files[1];
    return options;
}

void run(const std::vector<std::string> &args)
{
    const Options options = parseArguments(args);
    // The whole input is read and checked before the output file is touched.
    const auto samples = analyzer::augment(
        analyzer::parseFile(options.input, analyzer::LabelPolicy::Required), options.symmetries);
    std::ofstream out(options.output);

    if (!out)
        throw my_torch::Error("cannot write '" + options.output + "'");
    for (const analyzer::Sample &sample : samples)
        out << analyzer::formatFen(sample.position) << ' ' << analyzer::labelName(*sample.label)
            << '\n';
    out.flush();
    if (!out)
        throw my_torch::Error("cannot write '" + options.output + "'");
}

} // namespace

int main(int argc, char **argv)
{
    try {
        run({argv + 1, argv + argc});
    } catch (const std::exception &e) {
        std::cerr << "augment: " << e.what() << '\n' << usage;
        return EXIT_ERROR;
    }
    return 0;
}
