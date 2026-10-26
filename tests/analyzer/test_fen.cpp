#include <criterion/criterion.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "../../src/analyzer/Fen.hpp"
#include "my_torch/Error.hpp"

using namespace analyzer;

namespace {

constexpr const char *start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Real line from dataset/check/10_pieces.txt, explained in docs/reading-a-fen.md.
constexpr const char *guide = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White";

void rejects(const std::string &line)
{
    cr_assert_throw(parseLine(line, LabelPolicy::Optional), my_torch::Error, "accepted: %s",
                    line.c_str());
}

} // namespace

Test(fen, parses_the_starting_position)
{
    Position p = parseFen(start);

    cr_assert_eq(p.board[square(0, 8)], 'r');
    cr_assert_eq(p.board[square(4, 8)], 'k');
    cr_assert_eq(p.board[square(4, 1)], 'K');
    cr_assert_eq(p.board[square(3, 2)], 'P');
    cr_assert_eq(p.board[square(4, 4)], '.');
    cr_assert(p.sideToMove == Color::White);
    cr_assert(p.castling.whiteKingside && p.castling.whiteQueenside);
    cr_assert(p.castling.blackKingside && p.castling.blackQueenside);
    cr_assert(!p.enPassant.has_value());
    cr_assert_eq(p.halfmoveClock, 0);
    cr_assert_eq(p.fullmoveNumber, 1);
}

Test(fen, parses_the_guide_example)
{
    Sample s = parseLine(guide, LabelPolicy::Required);
    const Position &p = s.position;

    cr_assert_eq(p.board[square(0, 6)], 'R');
    cr_assert_eq(p.board[square(3, 6)], 'k');
    cr_assert_eq(p.board[square(4, 5)], 'r');
    cr_assert_eq(p.board[square(6, 5)], 'p');
    cr_assert_eq(p.board[square(5, 3)], 'K');
    cr_assert_eq(p.board[square(5, 2)], 'P');
    cr_assert(p.sideToMove == Color::Black);
    cr_assert(!p.castling.whiteKingside && !p.castling.blackQueenside);
    cr_assert_eq(p.halfmoveClock, 7);
    cr_assert_eq(p.fullmoveNumber, 59);
    cr_assert(s.label == Label::CheckWhite);
}

Test(fen, square_indices_follow_fen_order)
{
    cr_assert_eq(square(0, 8), 0);
    cr_assert_eq(square(7, 8), 7);
    cr_assert_eq(square(0, 1), 56);
    cr_assert_eq(square(7, 1), 63);
}

Test(fen, parses_en_passant_and_partial_castling)
{
    Position p = parseFen("4k3/8/8/8/4P3/8/8/4K2R b Kq e3 0 1");

    cr_assert(p.enPassant == square(4, 3));
    cr_assert(p.castling.whiteKingside && p.castling.blackQueenside);
    cr_assert(!p.castling.whiteQueenside && !p.castling.blackKingside);
}

Test(fen, all_labels_round_trip)
{
    for (Label label : {Label::Nothing, Label::CheckWhite, Label::CheckBlack, Label::CheckmateWhite,
                        Label::CheckmateBlack, Label::Stalemate})
        cr_assert(parseLabel(labelName(label)) == label);
    cr_assert(labelName(Label::CheckmateBlack) == "Checkmate Black");
}

Test(fen, label_is_optional_in_predict_mode)
{
    cr_assert(!parseLine(start, LabelPolicy::Optional).label.has_value());
    cr_assert_throw(parseLine(start, LabelPolicy::Required), my_torch::Error);
}

Test(fen, accepts_windows_line_endings_and_extra_spaces)
{
    Sample s = parseLine(std::string("8/8/R2k4/4r1p1/8/5K2/5P2/8  b - - 7 59  Check   White\r"),
                         LabelPolicy::Required);

    cr_assert(s.label == Label::CheckWhite);
}

Test(fen, rejects_malformed_lines)
{
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0");      // 5 fields
    rejects("rnbqkbnr/pppppppp/8/8/8/8/RNBQKBNR w KQkq - 0 1");             // 7 ranks
    rejects("rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");    // 9 squares
    rejects("rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");     // 7 squares
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1");   // 9 pieces
    rejects("rnbqkbnr/ppppXppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");    // unknown piece
    rejects("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");    // no black king
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w KQkq - 0 1");    // two white kings
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1");    // side to move
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KK - 0 1");      // duplicate castling
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Z - 0 1");       // unknown castling
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e5 0 1");   // en passant, rank
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z3 0 1");   // en passant, file
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1");   // halfmoves < 0
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - abc 1");  // halfmoves
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0");    // move number
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1x");   // suffix
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Check Red");
    rejects("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 check white");
}

Test(fen, parse_file_skips_blank_lines)
{
    const std::string path = "fen_test_valid.txt";
    std::ofstream(path) << guide << "\n\n" << start << " Nothing\n";

    auto samples = parseFile(path, LabelPolicy::Required);
    std::remove(path.c_str());
    cr_assert_eq(samples.size(), 2);
    cr_assert(samples[1].label == Label::Nothing);
}

Test(fen, parse_file_reports_the_line_number)
{
    const std::string path = "fen_test_invalid.txt";
    std::ofstream(path) << guide << "\n" << "8/8/8 w - - 0 1 Nothing\n";

    std::string message;
    try {
        parseFile(path, LabelPolicy::Required);
    } catch (const my_torch::Error &error) {
        message = error.what();
    }
    std::remove(path.c_str());
    cr_assert(message.find("fen_test_invalid.txt:2:") != std::string::npos, "%s", message.c_str());
}

Test(fen, parse_file_rejects_missing_file)
{
    cr_assert_throw(parseFile("does_not_exist.txt", LabelPolicy::Optional), my_torch::Error);
}

Test(fen, formats_the_starting_position)
{
    const std::string fen = formatFen(parseFen(start));

    cr_assert_str_eq(fen.c_str(), start);
}

Test(fen, formats_castling_and_en_passant)
{
    const char *fens[] = {
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 12 40",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59",
    };

    for (const char *fen : fens) {
        const std::string text = formatFen(parseFen(fen));

        cr_assert_str_eq(text.c_str(), fen);
    }
}

Test(fen, format_writes_a_dash_without_castling)
{
    const std::string fen = formatFen(parseFen("4k3/8/8/8/8/8/8/4K3 b - - 3 17"));

    cr_assert_str_eq(fen.c_str(), "4k3/8/8/8/8/8/8/4K3 b - - 3 17");
}
