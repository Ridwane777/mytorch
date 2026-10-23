#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Oracle.hpp"
#include "my_torch/Error.hpp"

namespace {

constexpr int EXIT_ERROR = 84;

struct Report {
    long lines = 0;
    long labelled = 0;
    long agree = 0;
    long stalemates = 0;
    long illegal = 0;
};

void check(const std::string &path, Report &report)
{
    const auto samples = analyzer::parseFile(path, analyzer::LabelPolicy::Optional);

    for (std::size_t i = 0; i < samples.size(); i++) {
        const oracle::Verdict verdict = oracle::classify(samples[i].position);
        report.lines++;
        if (verdict == oracle::Verdict::Stalemate)
            report.stalemates++;
        if (verdict == oracle::Verdict::Illegal)
            report.illegal++;
        if (!samples[i].label)
            continue;
        report.labelled++;
        if (oracle::toLabel(verdict) == samples[i].label) {
            report.agree++;
            continue;
        }
        // parseFile ignore les lignes vides : l'indice est celui de la position dans le fichier.
        std::cout << path << ": position " << i + 1 << ": expected "
                  << analyzer::labelName(*samples[i].label) << ", oracle "
                  << oracle::verdictName(verdict) << "\n";
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " CHESSFILE...\n";
        return EXIT_ERROR;
    }
    Report report;
    try {
        for (int i = 1; i < argc; i++)
            check(argv[i], report);
    } catch (const my_torch::Error &error) {
        std::cerr << error.what() << "\n";
        return EXIT_ERROR;
    }
    const double rate = report.labelled ? 100.0 * report.agree / report.labelled : 100.0;
    std::cout << "positions: " << report.lines << "\n"
              << "labelled: " << report.labelled << "\n"
              << "agree: " << report.agree << " (" << std::fixed << std::setprecision(2) << rate
              << " %)\n"
              << "disagree: " << report.labelled - report.agree << "\n"
              << "stalemates: " << report.stalemates << "\n"
              << "illegal: " << report.illegal << "\n";
    return 0;
}
