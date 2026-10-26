#include "Oracle.hpp"

#include <array>
#include <cctype>

#include "my_torch/Error.hpp"

namespace oracle {

namespace {

using analyzer::square;

struct Step {
    int file;
    int rank;
};

constexpr std::array<Step, 8> knightSteps = {
    {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};
constexpr std::array<Step, 8> kingSteps = {
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
constexpr std::array<Step, 4> straight = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr std::array<Step, 4> diagonal = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

// Squares are handled as (file 0-7, rank 1-8): a step outside these bounds leaves
// the board, whereas a 0-63 index would "overflow" from h4 to a3.
int fileOf(int sq)
{
    return sq % 8;
}

int rankOf(int sq)
{
    return 8 - sq / 8;
}

bool onBoard(int file, int rank)
{
    return file >= 0 && file < 8 && rank >= 1 && rank <= 8;
}

bool isEmpty(char piece)
{
    return piece == '.';
}

Color colorOf(char piece)
{
    return std::isupper(static_cast<unsigned char>(piece)) ? Color::White : Color::Black;
}

bool belongsTo(char piece, Color color)
{
    return !isEmpty(piece) && colorOf(piece) == color;
}

// FEN letter of a piece of the given colour ('N' for a white knight, 'n' for a black one).
char pieceOf(char type, Color color)
{
    return color == Color::White ? type : static_cast<char>(std::tolower(type));
}

char at(const Position &p, int file, int rank)
{
    return p.board[square(file, rank)];
}

// Does a sliding piece (rook, bishop, queen) attack the square from one of these directions?
template <std::size_t N>
bool slidingAttack(const Position &p, int file, int rank, const std::array<Step, N> &steps,
                   char rook, char queen)
{
    for (const Step s : steps)
        for (int f = file + s.file, r = rank + s.rank; onBoard(f, r); f += s.file, r += s.rank) {
            const char piece = at(p, f, r);
            if (isEmpty(piece))
                continue;
            if (piece == rook || piece == queen)
                return true;
            break;
        }
    return false;
}

void addPawnMoves(const Position &p, int from, std::vector<Move> &moves)
{
    const Color us = p.sideToMove;
    const int dir = us == Color::White ? 1 : -1;
    const int startRank = us == Color::White ? 2 : 7;
    const int lastRank = us == Color::White ? 8 : 1;
    const int file = fileOf(from);
    const int rank = rankOf(from);
    auto push = [&](int to, bool enPassant = false) {
        if (rankOf(to) != lastRank) {
            moves.push_back({from, to, 0, enPassant});
            return;
        }
        for (const char type : {'Q', 'R', 'B', 'N'})
            moves.push_back({from, to, pieceOf(type, us), false});
    };

    if (onBoard(file, rank + dir) && isEmpty(at(p, file, rank + dir))) {
        push(square(file, rank + dir));
        if (rank == startRank && isEmpty(at(p, file, rank + 2 * dir)))
            push(square(file, rank + 2 * dir));
    }
    for (const int df : {-1, 1}) {
        if (!onBoard(file + df, rank + dir))
            continue;
        const int to = square(file + df, rank + dir);
        if (belongsTo(p.board[to], opponent(us)))
            push(to);
        else if (p.enPassant == to)
            push(to, true);
    }
}

template <std::size_t N>
void addSteps(const Position &p, int from, const std::array<Step, N> &steps, bool slide,
              std::vector<Move> &moves)
{
    for (const Step s : steps)
        for (int f = fileOf(from) + s.file, r = rankOf(from) + s.rank; onBoard(f, r);
             f += s.file, r += s.rank) {
            const char target = at(p, f, r);
            if (belongsTo(target, p.sideToMove))
                break;
            moves.push_back({from, square(f, r)});
            if (!isEmpty(target) || !slide)
                break;
        }
}

// Moves that follow how the pieces move, without checking that the king stays safe.
std::vector<Move> pseudoLegalMoves(const Position &p)
{
    std::vector<Move> moves;

    for (int from = 0; from < 64; from++) {
        const char piece = p.board[from];
        if (!belongsTo(piece, p.sideToMove))
            continue;
        switch (std::toupper(static_cast<unsigned char>(piece))) {
        case 'P':
            addPawnMoves(p, from, moves);
            break;
        case 'N':
            addSteps(p, from, knightSteps, false, moves);
            break;
        case 'K':
            addSteps(p, from, kingSteps, false, moves);
            break;
        case 'R':
            addSteps(p, from, straight, true, moves);
            break;
        case 'B':
            addSteps(p, from, diagonal, true, moves);
            break;
        case 'Q':
            addSteps(p, from, straight, true, moves);
            addSteps(p, from, diagonal, true, moves);
            break;
        }
    }
    return moves;
}

} // namespace

Color opponent(Color color)
{
    return color == Color::White ? Color::Black : Color::White;
}

bool isAttacked(const Position &p, int sq, Color by)
{
    const int file = fileOf(sq);
    const int rank = rankOf(sq);
    // A white pawn attacks upwards: if it attacks this square, it is one rank lower.
    const int pawnRank = by == Color::White ? rank - 1 : rank + 1;

    for (const int df : {-1, 1})
        if (onBoard(file + df, pawnRank) && at(p, file + df, pawnRank) == pieceOf('P', by))
            return true;
    for (const Step s : knightSteps)
        if (onBoard(file + s.file, rank + s.rank) &&
            at(p, file + s.file, rank + s.rank) == pieceOf('N', by))
            return true;
    for (const Step s : kingSteps)
        if (onBoard(file + s.file, rank + s.rank) &&
            at(p, file + s.file, rank + s.rank) == pieceOf('K', by))
            return true;
    return slidingAttack(p, file, rank, straight, pieceOf('R', by), pieceOf('Q', by)) ||
           slidingAttack(p, file, rank, diagonal, pieceOf('B', by), pieceOf('Q', by));
}

std::vector<int> attackers(const Position &p, int sq, Color by)
{
    const int file = fileOf(sq);
    const int rank = rankOf(sq);
    const int pawnRank = by == Color::White ? rank - 1 : rank + 1;
    std::vector<int> found;
    const auto addIf = [&](int f, int r, char piece) {
        if (onBoard(f, r) && at(p, f, r) == piece)
            found.push_back(square(f, r));
    };

    for (const int df : {-1, 1})
        addIf(file + df, pawnRank, pieceOf('P', by));
    for (const Step s : knightSteps)
        addIf(file + s.file, rank + s.rank, pieceOf('N', by));
    for (const Step s : kingSteps)
        addIf(file + s.file, rank + s.rank, pieceOf('K', by));
    // Sliding pieces: the first piece met in each direction, if it moves along that direction.
    const auto slide = [&](const auto &steps, char line, char queen) {
        for (const Step s : steps)
            for (int f = file + s.file, r = rank + s.rank; onBoard(f, r);
                 f += s.file, r += s.rank) {
                const char piece = at(p, f, r);
                if (isEmpty(piece))
                    continue;
                if (piece == line || piece == queen)
                    found.push_back(square(f, r));
                break;
            }
    };
    slide(straight, pieceOf('R', by), pieceOf('Q', by));
    slide(diagonal, pieceOf('B', by), pieceOf('Q', by));
    return found;
}

int findKing(const Position &p, Color color)
{
    for (int sq = 0; sq < 64; sq++)
        if (p.board[sq] == pieceOf('K', color))
            return sq;
    throw my_torch::Error("no king on the board");
}

bool inCheck(const Position &p, Color color)
{
    return isAttacked(p, findKing(p, color), opponent(color));
}

// Castling rights are not updated: castling is never generated (ADR-0006).
Position play(const Position &p, const Move &move)
{
    Position next = p;
    const char piece = p.board[move.from];
    const bool pawn = std::toupper(static_cast<unsigned char>(piece)) == 'P';
    const bool capture = !isEmpty(p.board[move.to]) || move.enPassant;

    next.board[move.to] = move.promotion != 0 ? move.promotion : piece;
    next.board[move.from] = '.';
    if (move.enPassant)
        next.board[square(fileOf(move.to), rankOf(move.from))] = '.';
    next.enPassant.reset();
    const int advance = rankOf(move.to) - rankOf(move.from);
    if (pawn && (advance == 2 || advance == -2))
        next.enPassant = square(fileOf(move.from), (rankOf(move.from) + rankOf(move.to)) / 2);
    next.halfmoveClock = pawn || capture ? 0 : p.halfmoveClock + 1;
    if (p.sideToMove == Color::Black)
        next.fullmoveNumber++;
    next.sideToMove = opponent(p.sideToMove);
    return next;
}

std::vector<Move> legalMoves(const Position &p)
{
    std::vector<Move> legal;

    for (const Move &move : pseudoLegalMoves(p))
        if (!inCheck(play(p, move), p.sideToMove))
            legal.push_back(move);
    return legal;
}

Verdict classify(const Position &p)
{
    const Color mover = p.sideToMove;
    const Color attacker = opponent(mover);

    if (inCheck(p, attacker))
        return Verdict::Illegal;
    const bool check = inCheck(p, mover);
    const bool canMove = !legalMoves(p).empty();
    if (!check)
        return canMove ? Verdict::Nothing : Verdict::Stalemate;
    if (attacker == Color::White)
        return canMove ? Verdict::CheckWhite : Verdict::CheckmateWhite;
    return canMove ? Verdict::CheckBlack : Verdict::CheckmateBlack;
}

std::optional<analyzer::Label> toLabel(Verdict verdict)
{
    using analyzer::Label;

    switch (verdict) {
    case Verdict::Nothing:
        return Label::Nothing;
    case Verdict::CheckWhite:
        return Label::CheckWhite;
    case Verdict::CheckBlack:
        return Label::CheckBlack;
    case Verdict::CheckmateWhite:
        return Label::CheckmateWhite;
    case Verdict::CheckmateBlack:
        return Label::CheckmateBlack;
    case Verdict::Stalemate:
        return Label::Stalemate;
    default:
        return std::nullopt;
    }
}

const char *verdictName(Verdict verdict)
{
    switch (verdict) {
    case Verdict::Illegal:
        return "Illegal";
    default:
        return analyzer::labelName(*toLabel(verdict)).data();
    }
}

std::uint64_t perft(const Position &p, int depth)
{
    if (depth == 0)
        return 1;
    std::uint64_t count = 0;
    for (const Move &move : legalMoves(p))
        count += perft(play(p, move), depth - 1);
    return count;
}

} // namespace oracle
