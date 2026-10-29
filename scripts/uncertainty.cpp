// MC dropout analysis (#26, docs/performance/18-uncertainty.md): for every position of a labelled
// file, the plain prediction of the network and its Monte Carlo prediction (Network::
// predictMonteCarlo, ADR-0034), written one row per position for scripts/plot_uncertainty.py.
// Usage: uncertainty NETWORK LABELLED_FILE OUTPUT_CSV [PASSES (30)] [SEED (1)]
// Columns: expected,plain,plain_confidence,mc,mc_confidence,entropy,mutual_information,disagreement
// (class indices in the network's label order; confidences are the top probability).
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Cli.hpp"
#include "Encoding.hpp"
#include "Fen.hpp"
#include "my_torch/my_torch.hpp"

namespace {

std::size_t top(const my_torch::Matrix &m, std::size_t r)
{
    std::size_t best = 0;
    for (std::size_t c = 1; c < m.cols(); c++)
        if (m(r, c) > m(r, best))
            best = c;
    return best;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: uncertainty NETWORK LABELLED_FILE OUTPUT_CSV [PASSES] [SEED]\n";
        return 84;
    }
    try {
        my_torch::Network net = my_torch::Network::load(argv[1]);
        const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
        const auto &labels = net.info().labels;
        const std::vector<analyzer::Sample> samples =
            analyzer::parseFile(argv[2], analyzer::LabelPolicy::Required);
        const std::size_t passes = argc > 4 ? std::stoul(argv[4]) : 30;
        const std::uint64_t seed = argc > 5 ? std::stoull(argv[5]) : 1;
        std::ofstream out(argv[3]);
        out << "expected,plain,plain_confidence,mc,mc_confidence,entropy,mutual_information,"
               "disagreement\n";
        out.precision(6);
        constexpr std::size_t chunk = 4096;
        for (std::size_t begin = 0; begin < samples.size(); begin += chunk) {
            const std::size_t end = std::min(begin + chunk, samples.size());
            const my_torch::Matrix rows = analyzer::encodeRows(encoding, samples, begin, end);
            const my_torch::Matrix plain = net.predict(rows);
            // A seed per chunk, so that the result does not depend on the chunk size alone.
            const my_torch::UncertainPrediction mc =
                net.predictMonteCarlo(rows, passes, seed + begin);
            for (std::size_t r = 0; r < rows.rows(); r++) {
                const std::size_t expected = analyzer::labelIndex(
                    std::string(analyzer::labelName(*samples[begin + r].label)), labels);
                const std::size_t p = top(plain, r);
                const std::size_t m = top(mc.mean, r);
                out << expected << ',' << p << ',' << plain(r, p) << ',' << m << ','
                    << mc.mean(r, m) << ',' << mc.entropy[r] << ',' << mc.mutualInformation[r]
                    << ',' << mc.disagreement[r] << '\n';
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "uncertainty: " << e.what() << '\n';
        return 84;
    }
    return 0;
}
