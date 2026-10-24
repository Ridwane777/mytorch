#include "Positions.hpp"

#include <cmath>

#include "my_torch/Error.hpp"

namespace positions {

namespace {

// SplitMix64 step: portable, so a seed gives the same positions on every toolchain.
std::uint64_t next(std::uint64_t &state)
{
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint64_t below(std::uint64_t &state, std::uint64_t range)
{
    return next(state) % range; // bias < 2^-50 for these tiny ranges
}

int emptySquare(std::uint64_t &state, const Position &p, bool pawn)
{
    while (true) {
        const int sq = static_cast<int>(below(state, 64));
        const int rank = 8 - sq / 8;
        if (p.board[static_cast<std::size_t>(sq)] == '.' && !(pawn && (rank == 1 || rank == 8)))
            return sq;
    }
}

} // namespace

std::string toFen(const Position &p)
{
    std::string fen;
    for (int rank = 0; rank < 8; rank++) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            const char c = p.board[static_cast<std::size_t>(rank * 8 + file)];
            if (c == '.') {
                empty++;
                continue;
            }
            if (empty)
                fen += static_cast<char>('0' + empty);
            empty = 0;
            fen += c;
        }
        if (empty)
            fen += static_cast<char>('0' + empty);
        if (rank < 7)
            fen += '/';
    }
    fen += p.sideToMove == analyzer::Color::White ? " w" : " b";
    std::string castling;
    if (p.castling.whiteKingside)
        castling += 'K';
    if (p.castling.whiteQueenside)
        castling += 'Q';
    if (p.castling.blackKingside)
        castling += 'k';
    if (p.castling.blackQueenside)
        castling += 'q';
    fen += ' ' + (castling.empty() ? "-" : castling) + ' ';
    if (p.enPassant)
        fen += std::string{static_cast<char>('a' + *p.enPassant % 8),
                           static_cast<char>('0' + 8 - *p.enPassant / 8)};
    else
        fen += '-';
    return fen + ' ' + std::to_string(p.halfmoveClock) + ' ' + std::to_string(p.fullmoveNumber);
}

Position randomBoard(std::uint64_t &state, int pieces)
{
    static constexpr char types[] = {'Q', 'R', 'B', 'N', 'P'};
    Position p;
    p.board.fill('.');
    p.board[static_cast<std::size_t>(emptySquare(state, p, false))] = 'K';
    p.board[static_cast<std::size_t>(emptySquare(state, p, false))] = 'k';
    for (int i = 0; i < pieces; i++) {
        const char type = types[below(state, 5)];
        const bool white = below(state, 2) == 0;
        p.board[static_cast<std::size_t>(emptySquare(state, p, type == 'P'))] =
            white ? type : static_cast<char>(type - 'A' + 'a');
    }
    p.sideToMove = below(state, 2) == 0 ? analyzer::Color::White : analyzer::Color::Black;
    return p;
}

std::vector<Labeled> generate(std::uint64_t count, std::uint64_t seed, double stalemateShare,
                              Report *report)
{
    if (!(stalemateShare >= 0 && stalemateShare <= 1))
        throw my_torch::Error("stalemate share must be in [0, 1]");
    const auto wanted = static_cast<std::uint64_t>(std::llround(count * stalemateShare));
    std::uint64_t state = seed;
    std::vector<Labeled> stalemates;
    std::vector<Labeled> others;
    Report r;

    while (stalemates.size() < wanted || others.size() < count - wanted) {
        const int pieces = 1 + static_cast<int>(below(state, 6));
        const Position p = randomBoard(state, pieces);
        r.drawn++;
        const oracle::Verdict verdict = oracle::classify(p);
        if (verdict == oracle::Verdict::Illegal)
            continue;
        r.legal++;
        if (verdict == oracle::Verdict::Stalemate) {
            r.stalemates++;
            if (stalemates.size() < wanted)
                stalemates.push_back({p, verdict});
        } else if (others.size() < count - wanted) {
            others.push_back({p, verdict});
        }
    }
    // Interleave the two groups in a seeded order, so a file is not sorted by label.
    std::vector<Labeled> all = std::move(stalemates);
    all.insert(all.end(), others.begin(), others.end());
    for (std::size_t i = all.size(); i > 1; i--)
        std::swap(all[i - 1], all[below(state, i)]);
    if (report)
        *report = r;
    return all;
}

} // namespace positions
