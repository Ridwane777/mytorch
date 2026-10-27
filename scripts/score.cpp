// Scores predictions with the library's metrics (ADR-0026): the numbers published in
// docs/performance/ come from my_torch::report, the tested code, not from a second implementation.
//
//   score NETWORK EXPECTED PREDICTIONS OUTPUT_DIRECTORY [--families]
//
// NETWORK gives the class names (its labels, in order). EXPECTED has one "FEN label" position per
// line (the label is everything after the six FEN fields); PREDICTIONS has one label per line, in
// the same order, as written by my_torch_analyzer --predict. Blank lines are ignored in both.
// Writes to OUTPUT_DIRECTORY:
//   scores.txt     positions, accuracy, macro F1, then precision, recall, F1 and support per class;
//   confusion.csv  expected (rows) against predicted (columns);
//   report.json    the whole report (my_torch::writeReportJson);
//   families.json  with --families: the same report after merging the classes whose names share
//                  their first word ("Check White" and "Check Black" become "Check").
// Errors (unknown label, different line counts, unreadable file) go to stderr, exit code 84.
// Build (also done by the scripts that use it):
//   make build/libmy_torch.a
//   g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/score.cpp
//       build/libmy_torch.a -o score
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

namespace {

constexpr int EXIT_ERROR = 84;

constexpr const char *usage =
    "usage: score NETWORK EXPECTED PREDICTIONS OUTPUT_DIRECTORY [--families]\n";

std::vector<std::string> nonBlankLines(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw my_torch::Error("cannot open '" + path + "'");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        if (line.find_first_not_of(" \t\r") != std::string::npos)
            lines.push_back(line);
    return lines;
}

// The words of `text` from index `first` on, joined by single spaces.
std::string wordsFrom(const std::string &text, std::size_t first)
{
    std::istringstream words(text);
    std::string word;
    std::string out;
    for (std::size_t i = 0; words >> word; i++)
        if (i >= first)
            out += (out.empty() ? "" : " ") + word;
    return out;
}

std::size_t classOf(const std::vector<std::string> &labels, const std::string &label,
                    const std::string &where)
{
    for (std::size_t c = 0; c < labels.size(); c++)
        if (labels[c] == label)
            return c;
    throw my_torch::Error(where + ": unknown label '" + label + "'");
}

std::string fixed4(my_torch::Scalar value)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.4f", value);
    return text;
}

std::ofstream openOutput(const std::string &path)
{
    std::ofstream out(path);
    if (!out)
        throw my_torch::Error("cannot write '" + path + "'");
    return out;
}

void writeScores(const my_torch::ClassificationReport &rep, const std::vector<std::string> &labels,
                 const std::string &path)
{
    std::ofstream out = openOutput(path);
    out << "positions scored: " << rep.confusion.total() << '\n'
        << "accuracy: " << fixed4(rep.accuracy) << '\n'
        << "macro F1: " << fixed4(rep.macroF1) << '\n';
    for (std::size_t c = 0; c < labels.size(); c++) {
        const my_torch::ClassMetrics &m = rep.perClass[c];
        out << labels[c] << ": precision " << fixed4(m.precision) << ", recall "
            << fixed4(m.recall) << ", F1 " << fixed4(m.f1) << ", support " << m.support << '\n';
    }
}

void writeConfusion(const my_torch::ConfusionMatrix &m, const std::vector<std::string> &labels,
                    const std::string &path)
{
    std::ofstream out = openOutput(path);
    out << "expected \\ predicted";
    for (const std::string &label : labels)
        out << ',' << label;
    out << '\n';
    for (std::size_t e = 0; e < labels.size(); e++) {
        out << labels[e];
        for (std::size_t p = 0; p < labels.size(); p++)
            out << ',' << m.counts[e][p];
        out << '\n';
    }
}

void writeFamilies(const my_torch::ConfusionMatrix &m, const std::vector<std::string> &labels,
                   const std::string &path)
{
    std::vector<std::string> families;
    std::vector<std::size_t> groupOf;
    for (const std::string &label : labels) {
        const std::string family = label.substr(0, label.find(' '));
        std::size_t g = 0;
        while (g < families.size() && families[g] != family)
            g++;
        if (g == families.size())
            families.push_back(family);
        groupOf.push_back(g);
    }
    my_torch::writeReportJson(my_torch::report(my_torch::merge(m, groupOf, families.size())), path,
                              families);
}

void run(int argc, char **argv)
{
    const bool families = argc == 6 && std::string(argv[5]) == "--families";
    if (argc != 5 && !families)
        throw my_torch::Error(std::string("expected 4 arguments\n") + usage);
    const std::vector<std::string> labels = my_torch::Network::load(argv[1]).info().labels;
    const std::vector<std::string> expectedLines = nonBlankLines(argv[2]);
    const std::vector<std::string> predictedLines = nonBlankLines(argv[3]);
    if (expectedLines.size() != predictedLines.size())
        throw my_torch::Error(std::to_string(expectedLines.size()) + " positions in '" + argv[2] +
                              "' but " + std::to_string(predictedLines.size()) +
                              " predictions in '" + argv[3] + "'");
    std::vector<std::size_t> expected;
    std::vector<std::size_t> predicted;
    for (std::size_t i = 0; i < expectedLines.size(); i++) {
        const std::string line = std::to_string(i + 1);
        expected.push_back(
            classOf(labels, wordsFrom(expectedLines[i], 6), std::string(argv[2]) + ":" + line));
        predicted.push_back(
            classOf(labels, wordsFrom(predictedLines[i], 0), std::string(argv[3]) + ":" + line));
    }
    const my_torch::ConfusionMatrix matrix = my_torch::confusion(expected, predicted, labels.size());
    const my_torch::ClassificationReport rep = my_torch::report(matrix);
    const std::string dir = std::string(argv[4]) + "/";
    writeScores(rep, labels, dir + "scores.txt");
    writeConfusion(matrix, labels, dir + "confusion.csv");
    my_torch::writeReportJson(rep, dir + "report.json", labels);
    if (families)
        writeFamilies(matrix, labels, dir + "families.json");
}

} // namespace

int main(int argc, char **argv)
{
    try {
        run(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "score: " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
