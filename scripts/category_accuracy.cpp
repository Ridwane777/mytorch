// Accuracy of a network's predictions per category of position (#18): the piece giving check, the
// number of pieces on the board, the side in check. The categories come from the rule-based oracle
// (bonus/oracle, #37), never from the network.
// Usage: category_accuracy LABELLED_FILE PREDICTIONS OUTPUT_CSV
//   LABELLED_FILE: "FEN expected" per line (a split file); PREDICTIONS: one label per line, in the
//   same order (the output of my_torch_analyzer --predict). Writes
//   group,category,positions,correct,accuracy.
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Fen.hpp"
#include "Oracle.hpp"

namespace {

using analyzer::Color;
using analyzer::Position;

const char *pieceName(char letter)
{
    switch (letter | 0x20) {
    case 'p':
        return "pawn";
    case 'n':
        return "knight";
    case 'b':
        return "bishop";
    case 'r':
        return "rook";
    case 'q':
        return "queen";
    default:
        return "king";
    }
}

// The piece giving check to whichever king is attacked, "double check", or "none".
std::string checker(const Position &p)
{
    for (const Color color : {Color::White, Color::Black}) {
        const int king = oracle::findKing(p, color);
        if (king < 0)
            continue;
        const std::vector<int> from = oracle::attackers(p, king, oracle::opponent(color));
        if (from.size() >= 2)
            return "double check";
        if (from.size() == 1)
            return pieceName(p.board[static_cast<std::size_t>(from[0])]);
    }
    return "none";
}

std::string sideInCheck(const Position &p)
{
    if (oracle::inCheck(p, Color::White))
        return "white in check";
    if (oracle::inCheck(p, Color::Black))
        return "black in check";
    return "no check";
}

std::string pieceCount(const Position &p)
{
    int count = 0;
    for (const char c : p.board)
        count += c != '.';
    static const std::vector<std::pair<int, const char *>> bins = {
        {5, "02-05"}, {10, "06-10"}, {15, "11-15"}, {20, "16-20"}, {25, "21-25"}, {32, "26-32"}};
    for (const auto &[top, name] : bins)
        if (count <= top)
            return name;
    return "26-32";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: category_accuracy LABELLED_FILE PREDICTIONS OUTPUT_CSV\n";
        return 84;
    }
    try {
        const std::vector<analyzer::Sample> samples =
            analyzer::parseFile(argv[1], analyzer::LabelPolicy::Required);
        std::ifstream predictions(argv[2]);
        // group -> category -> {positions, correct}
        std::map<std::string, std::map<std::string, std::pair<long, long>>> table;
        std::string predicted;
        for (const analyzer::Sample &s : samples) {
            if (!std::getline(predictions, predicted))
                throw std::runtime_error("fewer predictions than positions");
            const bool right = predicted == analyzer::labelName(*s.label);
            for (const auto &[group, category] :
                 {std::pair<std::string, std::string>{"checking piece", checker(s.position)},
                  {"pieces on the board", pieceCount(s.position)},
                  {"side in check", sideInCheck(s.position)}}) {
                auto &cell = table[group][category];
                cell.first++;
                cell.second += right;
            }
        }
        if (std::getline(predictions, predicted))
            throw std::runtime_error("more predictions than positions");
        std::ofstream out(argv[3]);
        out << "group,category,positions,correct,accuracy\n";
        for (const auto &[group, categories] : table)
            for (const auto &[category, cell] : categories)
                out << group << ',' << category << ',' << cell.first << ',' << cell.second << ','
                    << static_cast<double>(cell.second) / static_cast<double>(cell.first) << '\n';
    } catch (const std::exception &e) {
        std::cerr << "category_accuracy: " << e.what() << '\n';
        return 84;
    }
    return 0;
}
