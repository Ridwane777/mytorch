// Exploratory analysis of the dataset (docs/performance/02-eda.md): one pass over every position,
// with the oracle to tell which piece gives check. Writes six CSV files to OUTPUT.
// Usage: my_torch_eda DATASET_DIRECTORY OUTPUT_DIRECTORY
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Oracle.hpp"
#include "my_torch/Error.hpp"

namespace fs = std::filesystem;
using analyzer::Color;
using analyzer::Label;

namespace {

constexpr std::string_view pieces = "PNBRQKpnbrqk";
constexpr std::size_t classes = 5;

int value(char piece)
{
    switch (std::toupper(static_cast<unsigned char>(piece))) {
    case 'P':
        return 1;
    case 'N':
    case 'B':
        return 3;
    case 'R':
        return 5;
    case 'Q':
        return 9;
    default:
        return 0;
    }
}

struct ClassStats {
    std::size_t positions = 0;
    std::array<std::size_t, 12> pieceTotal{};   // pieces of each kind over all positions
    std::array<std::size_t, 12> pieceAny{};     // positions holding at least one
    std::array<std::size_t, 64> kingSquare{};   // square of the king of the side to move
    std::map<int, std::size_t> material;        // attacker minus defender material
    std::array<std::size_t, 6> checker{};       // type of the attacking piece, P N B R Q K
    std::size_t doubleChecks = 0;
    std::size_t noAttacker = 0;                 // check classes without an attacker (should be 0)
    std::size_t castling = 0;
    std::size_t enPassant = 0;
    std::size_t halfmoveSum = 0;
    std::size_t fullmoveSum = 0;
};

struct FileStats {
    std::array<std::size_t, classes> lines{};
    std::size_t pieceSum = 0;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " DATASET_DIRECTORY OUTPUT_DIRECTORY\n";
        return 84;
    }
    try {
        std::vector<fs::path> files;
        for (const auto &dir : fs::directory_iterator(argv[1]))
            if (dir.is_directory())
                for (const auto &file : fs::directory_iterator(dir.path()))
                    if (file.path().extension() == ".txt")
                        files.push_back(file.path());
        std::sort(files.begin(), files.end());
        if (files.empty())
            throw my_torch::Error(std::string("no .txt file under ") + argv[1]);

        std::array<ClassStats, classes> stats;
        std::map<std::string, FileStats> perFile;
        for (const fs::path &file : files) {
            const std::string name = file.parent_path().filename().string() + "/" +
                                     file.filename().string();
            for (const auto &sample :
                 analyzer::parseFile(file.string(), analyzer::LabelPolicy::Required)) {
                const auto &p = sample.position;
                const auto label = static_cast<std::size_t>(*sample.label);
                if (label >= classes)
                    continue;
                ClassStats &s = stats[label];
                s.positions++;
                std::size_t count = 0;
                std::array<int, 2> material{}; // white, black
                for (std::size_t sq = 0; sq < 64; sq++) {
                    const std::size_t kind = pieces.find(p.board[sq]);
                    if (kind == std::string_view::npos)
                        continue;
                    count++;
                    s.pieceTotal[kind]++;
                    material[kind < 6 ? 0 : 1] += value(p.board[sq]);
                }
                for (std::size_t kind = 0; kind < 12; kind++)
                    s.pieceAny[kind] +=
                        std::count(p.board.begin(), p.board.end(), pieces[kind]) > 0;
                const Color mover = p.sideToMove;
                const Color other = oracle::opponent(mover);
                const int king = oracle::findKing(p, mover);
                s.kingSquare[static_cast<std::size_t>(king)]++;
                const int attackerMaterial = material[other == Color::White ? 0 : 1];
                const int defenderMaterial = material[mover == Color::White ? 0 : 1];
                s.material[attackerMaterial - defenderMaterial]++;
                if (label != static_cast<std::size_t>(Label::Nothing)) {
                    const auto found = oracle::attackers(p, king, other);
                    if (found.empty())
                        s.noAttacker++;
                    if (found.size() >= 2)
                        s.doubleChecks++;
                    for (const int sq : found) {
                        const auto cell = static_cast<unsigned char>(
                            p.board[static_cast<std::size_t>(sq)]);
                        const char piece = static_cast<char>(std::toupper(cell));
                        s.checker[std::string_view("PNBRQK").find(piece)]++;
                    }
                }
                s.castling += p.castling.whiteKingside || p.castling.whiteQueenside ||
                              p.castling.blackKingside || p.castling.blackQueenside;
                s.enPassant += p.enPassant.has_value();
                s.halfmoveSum += static_cast<std::size_t>(p.halfmoveClock);
                s.fullmoveSum += static_cast<std::size_t>(p.fullmoveNumber);
                perFile[name].lines[label]++;
                perFile[name].pieceSum += count;
            }
        }

        const fs::path out(argv[2]);
        fs::create_directories(out);
        const auto labelOf = [](std::size_t c) {
            return std::string(analyzer::labelName(static_cast<Label>(c)));
        };
        std::ofstream checkers(out / "eda_checkers.csv");
        checkers << "label,positions,pawn,knight,bishop,rook,queen,king,double_checks,"
                    "no_attacker\n";
        std::ofstream piecesFile(out / "eda_pieces.csv");
        piecesFile << "label,piece,mean_count,share_with_at_least_one\n";
        std::ofstream kings(out / "eda_kings.csv");
        kings << "label,square,file,rank,count\n";
        std::ofstream materialFile(out / "eda_material.csv");
        materialFile << "label,attacker_minus_defender,count\n";
        std::ofstream fen(out / "eda_fen.csv");
        fen << "label,positions,share_with_castling_rights,share_with_en_passant,mean_halfmove,"
               "mean_fullmove\n";
        for (std::size_t c = 0; c < classes; c++) {
            const ClassStats &s = stats[c];
            const double n = static_cast<double>(s.positions);
            checkers << labelOf(c) << ',' << s.positions;
            for (const std::size_t k : s.checker)
                checkers << ',' << k;
            checkers << ',' << s.doubleChecks << ',' << s.noAttacker << '\n';
            for (std::size_t k = 0; k < 12; k++)
                piecesFile << labelOf(c) << ',' << pieces[k] << ',' << s.pieceTotal[k] / n << ','
                           << s.pieceAny[k] / n << '\n';
            for (std::size_t sq = 0; sq < 64; sq++)
                kings << labelOf(c) << ',' << sq << ',' << "abcdefgh"[sq % 8] << ',' << 8 - sq / 8
                      << ',' << s.kingSquare[sq] << '\n';
            for (const auto &[balance, count] : s.material)
                materialFile << labelOf(c) << ',' << balance << ',' << count << '\n';
            fen << labelOf(c) << ',' << s.positions << ',' << s.castling / n << ','
                << s.enPassant / n << ',' << s.halfmoveSum / n << ',' << s.fullmoveSum / n << '\n';
        }
        std::ofstream filesOut(out / "eda_files.csv");
        filesOut << "file,lines,nothing,check_white,check_black,checkmate_white,checkmate_black,"
                    "mean_pieces\n";
        for (const auto &[name, f] : perFile) {
            std::size_t total = 0;
            for (const std::size_t l : f.lines)
                total += l;
            filesOut << name << ',' << total;
            for (const std::size_t l : f.lines)
                filesOut << ',' << l;
            filesOut << ',' << static_cast<double>(f.pieceSum) / static_cast<double>(total) << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << argv[0] << ": " << e.what() << '\n';
        return 84;
    }
    return 0;
}
