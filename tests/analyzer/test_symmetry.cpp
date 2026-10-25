#include <criterion/criterion.h>

#include <string>
#include <vector>

#include "../../src/analyzer/Fen.hpp"
#include "../../src/analyzer/Symmetry.hpp"

using namespace analyzer;

namespace {

// Expected positions are written by hand, not computed with the code under test.
constexpr const char *afterE4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
constexpr const char *afterE4Mirror = "rnbkqbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBKQBNR b - d3 0 1";
constexpr const char *afterE4Flip = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1";
constexpr const char *afterE4Both = "rnbkqbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBKQBNR w - d6 0 1";

// Black king on e8 attacked by the white rook on e1: White gives check.
constexpr const char *rookCheck = "4k3/8/8/8/8/8/8/4RK2 b - - 0 1";
constexpr const char *rookCheckMirror = "3k4/8/8/8/8/8/8/2KR4 b - - 0 1";
constexpr const char *rookCheckFlip = "4rk2/8/8/8/8/8/8/4K3 w - - 0 1";

// Back-rank mate: the white rook on a8 mates the black king on g8.
constexpr const char *backRank = "R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1";
constexpr const char *backRankMirror = "1k5R/ppp5/8/8/8/8/8/3K4 b - - 0 1";
constexpr const char *backRankFlip = "4k3/8/8/8/8/8/5PPP/r5K1 w - - 0 1";
constexpr const char *backRankBoth = "3k4/8/8/8/8/8/PPP5/1K5r w - - 0 1";

std::string fenOf(const Position &position)
{
    return formatFen(position);
}

void assertFen(const Position &position, const char *expected)
{
    const std::string fen = fenOf(position);

    cr_assert_str_eq(fen.c_str(), expected);
}

} // namespace

Test(symmetry, flip_twice_is_the_identity)
{
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        afterE4,
        "r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 12 40",
        "r3k2r/p6p/8/8/8/8/P6P/R3K2R w KQ - 5 20",
        "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59",
        backRank,
    };

    for (const char *fen : fens) {
        const std::string twice = fenOf(flipColors(flipColors(parseFen(fen))));

        cr_assert_str_eq(twice.c_str(), fen);
    }
}

Test(symmetry, mirror_twice_is_the_identity_without_castling)
{
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - e3 0 1",
        "r3k2r/8/8/8/3pP3/8/8/R3K2R b - e3 12 40",
        "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59",
        backRank,
    };

    for (const char *fen : fens) {
        const std::string twice = fenOf(mirrorFiles(mirrorFiles(parseFen(fen))));

        cr_assert_str_eq(twice.c_str(), fen);
    }
}

Test(symmetry, mirror_clears_every_castling_right)
{
    const Position mirrored =
        mirrorFiles(parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    cr_assert(!mirrored.castling.whiteKingside && !mirrored.castling.whiteQueenside);
    cr_assert(!mirrored.castling.blackKingside && !mirrored.castling.blackQueenside);
    assertFen(mirrored, "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1");
}

Test(symmetry, mirror_moves_the_en_passant_file_and_keeps_the_side)
{
    const Position mirrored = mirrorFiles(parseFen(afterE4));

    cr_assert_eq(mirrored.enPassant.value(), square(3, 3)); // e3 becomes d3
    cr_assert(mirrored.sideToMove == Color::Black);
    assertFen(mirrored, afterE4Mirror);
}

Test(symmetry, mirror_keeps_the_clocks_and_the_pieces_letters)
{
    assertFen(mirrorFiles(parseFen("r3k2r/8/8/8/3pP3/8/8/R3K2R b - e3 12 40")),
              "r2k3r/8/8/8/3Pp3/8/8/R2K3R b - d3 12 40");
}

Test(symmetry, flip_moves_the_en_passant_square_and_swaps_the_side)
{
    const Position flipped = flipColors(parseFen(afterE4));

    cr_assert_eq(flipped.enPassant.value(), square(4, 6)); // e3 becomes e6
    cr_assert(flipped.sideToMove == Color::White);
    assertFen(flipped, afterE4Flip);
}

Test(symmetry, flip_swaps_the_castling_rights_and_keeps_the_clocks)
{
    const Position flipped = flipColors(parseFen("r3k2r/8/8/8/3pP3/8/8/R3K2R b Kq e3 12 40"));

    cr_assert(flipped.castling.whiteQueenside && flipped.castling.blackKingside);
    cr_assert(!flipped.castling.whiteKingside && !flipped.castling.blackQueenside);
    assertFen(flipped, "r3k2r/8/8/3Pp3/8/8/8/R3K2R w Qk e6 12 40");
}

Test(symmetry, flip_swaps_the_colour_of_check_and_checkmate_only)
{
    cr_assert(flipLabel(Label::Nothing) == Label::Nothing);
    cr_assert(flipLabel(Label::CheckWhite) == Label::CheckBlack);
    cr_assert(flipLabel(Label::CheckBlack) == Label::CheckWhite);
    cr_assert(flipLabel(Label::CheckmateWhite) == Label::CheckmateBlack);
    cr_assert(flipLabel(Label::CheckmateBlack) == Label::CheckmateWhite);
    cr_assert(flipLabel(Label::Stalemate) == Label::Stalemate);
}

Test(symmetry, rook_check_flips_to_the_other_colour_and_survives_the_mirror)
{
    const Sample sample = parseLine(std::string(rookCheck) + " Check White", LabelPolicy::Required);
    const Sample flipped = flipColors(sample);
    const Sample mirrored = mirrorFiles(sample);

    assertFen(flipped.position, rookCheckFlip);
    cr_assert(flipped.label == Label::CheckBlack);
    assertFen(mirrored.position, rookCheckMirror);
    cr_assert(mirrored.label == Label::CheckWhite);
}

Test(symmetry, back_rank_mate_flips_to_the_other_colour_and_survives_the_mirror)
{
    const Sample sample =
        parseLine(std::string(backRank) + " Checkmate White", LabelPolicy::Required);
    const Sample flipped = flipColors(sample);
    const Sample mirrored = mirrorFiles(sample);

    assertFen(flipped.position, backRankFlip);
    cr_assert(flipped.label == Label::CheckmateBlack);
    assertFen(mirrored.position, backRankMirror);
    cr_assert(mirrored.label == Label::CheckmateWhite);
}

Test(symmetry, a_sample_without_label_stays_without_label)
{
    const Sample sample = parseLine(rookCheck, LabelPolicy::Optional);

    cr_assert(!mirrorFiles(sample).label.has_value());
    cr_assert(!flipColors(sample).label.has_value());
}

Test(symmetry, augment_all_gives_four_samples_per_input_in_order)
{
    const std::vector<Sample> input = {
        parseLine(std::string(afterE4) + " Nothing", LabelPolicy::Required),
        parseLine(std::string(backRank) + " Checkmate White", LabelPolicy::Required),
    };
    const auto out = augment(input, Symmetries::All);

    cr_assert_eq(out.size(), 8);
    assertFen(out[0].position, afterE4);
    assertFen(out[1].position, afterE4Mirror);
    assertFen(out[2].position, afterE4Flip);
    assertFen(out[3].position, afterE4Both);
    for (int i = 0; i < 4; i++)
        cr_assert(out[i].label == Label::Nothing);
    assertFen(out[4].position, backRank);
    assertFen(out[5].position, backRankMirror);
    assertFen(out[6].position, backRankFlip);
    assertFen(out[7].position, backRankBoth);
    cr_assert(out[4].label == Label::CheckmateWhite);
    cr_assert(out[5].label == Label::CheckmateWhite);
    cr_assert(out[6].label == Label::CheckmateBlack);
    cr_assert(out[7].label == Label::CheckmateBlack);
}

Test(symmetry, augment_keeps_only_the_enabled_symmetries)
{
    const std::vector<Sample> input = {
        parseLine(std::string(rookCheck) + " Check White", LabelPolicy::Required)};
    const auto none = augment(input, Symmetries::None);
    const auto mirror = augment(input, Symmetries::Mirror);
    const auto flip = augment(input, Symmetries::Flip);

    cr_assert_eq(none.size(), 1);
    cr_assert_eq(mirror.size(), 2);
    assertFen(mirror[1].position, rookCheckMirror);
    cr_assert_eq(flip.size(), 2);
    assertFen(flip[1].position, rookCheckFlip);
    cr_assert(flip[1].label == Label::CheckBlack);
    cr_assert(augment({}, Symmetries::All).empty());
}

Test(symmetry, augment_keeps_the_stalemate_label)
{
    // Real stalemate: the black king on a8 is not attacked and a7, b7 and b8 are covered.
    const std::vector<Sample> input = {
        parseLine("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1 Stalemate", LabelPolicy::Required)};
    const auto out = augment(input, Symmetries::All);

    cr_assert_eq(out.size(), 4);
    for (const Sample &sample : out)
        cr_assert(sample.label == Label::Stalemate);
}
