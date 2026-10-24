#include <charconv>
#include <iostream>
#include <string>

#include "Positions.hpp"
#include "my_torch/Error.hpp"

namespace {

constexpr int EXIT_ERROR = 84;

std::uint64_t parseCount(const char *text)
{
    const std::string s(text);
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{} || end != s.data() + s.size())
        throw my_torch::Error("'" + s + "' is not a non-negative integer");
    return value;
}

} // namespace

// Usage: my_torch_positions COUNT SEED [STALEMATE_SHARE]
// Writes COUNT labeled positions ("FEN label", the dataset format) on standard output and the
// generation statistics on standard error.
int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << argv[0] << " COUNT SEED [STALEMATE_SHARE (default 0.5)]\n";
        return EXIT_ERROR;
    }
    try {
        const double share = argc == 4 ? std::stod(argv[3]) : 0.5;
        positions::Report report;
        for (const auto &l : positions::generate(parseCount(argv[1]), parseCount(argv[2]), share,
                                                 &report))
            std::cout << positions::toFen(l.position) << ' ' << oracle::verdictName(l.verdict)
                      << '\n';
        std::cerr << "drawn " << report.drawn << ", legal " << report.legal << ", stalemates "
                  << report.stalemates << '\n';
    } catch (const std::exception &e) {
        std::cerr << argv[0] << ": " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
