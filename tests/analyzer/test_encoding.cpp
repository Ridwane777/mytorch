#include <criterion/criterion.h>

#include <string>
#include <vector>

#include "../../src/analyzer/Encoding.hpp"
#include "../../src/analyzer/Fen.hpp"
#include "my_torch/Error.hpp"

using namespace analyzer;

namespace {

constexpr const char *start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Two kings only: Black to move, no castling rights.
constexpr const char *kingsBlack = "4k3/8/8/8/8/8/8/4K3 b - - 0 1";

std::vector<my_torch::Scalar> encode(const std::string &name, const std::string &fen)
{
    const Encoding &encoding = findEncoding(name);
    std::vector<my_torch::Scalar> row(encoding.width(), 0);
    encoding.encode(parseFen(fen), row.data());
    return row;
}

int countSet(const std::vector<my_torch::Scalar> &row, std::size_t begin, std::size_t end)
{
    int count = 0;
    for (std::size_t i = begin; i < end; i++)
        count += row[i] != 0 ? 1 : 0;
    return count;
}

// Attack input of a square: [773, 837) for White, [837, 901) for Black.
bool whiteAttacks(const std::vector<my_torch::Scalar> &row, int file, int rank)
{
    return row[773 + static_cast<std::size_t>(square(file, rank))] == 1;
}

bool blackAttacks(const std::vector<my_torch::Scalar> &row, int file, int rank)
{
    return row[837 + static_cast<std::size_t>(square(file, rank))] == 1;
}

} // namespace

Test(encoding, finds_the_four_encodings_with_the_width_in_their_name)
{
    for (const std::string name :
         {"fen-onehot-768", "fen-onehot-773", "fen-signed-385", "fen-attacks-901"}) {
        const Encoding &encoding = findEncoding(name);
        cr_assert(encoding.name() == name);
        cr_assert(std::to_string(encoding.width()) == name.substr(name.rfind('-') + 1));
    }
}

Test(encoding, an_unknown_name_lists_the_four_known_ones)
{
    try {
        findEncoding("fen-nothing-1");
        cr_assert_fail("accepted an unknown encoding");
    } catch (const my_torch::Error &error) {
        const std::string message = error.what();
        cr_assert(message.find("'fen-nothing-1'") != std::string::npos, "%s", message.c_str());
        for (const char *name :
             {"fen-onehot-768", "fen-onehot-773", "fen-signed-385", "fen-attacks-901"})
            cr_assert(message.find(name) != std::string::npos, "%s missing: %s", name,
                      message.c_str());
    }
}

Test(encoding, onehot_768_keeps_its_layout)
{
    const auto row = encode("fen-onehot-768", start);

    cr_assert_eq(row.size(), 768);
    cr_assert_eq(countSet(row, 0, 768), 32);
    cr_assert_eq(row[0 * 64 + static_cast<std::size_t>(square(4, 2))], 1);  // P on e2
    cr_assert_eq(row[5 * 64 + static_cast<std::size_t>(square(4, 1))], 1);  // K on e1
    cr_assert_eq(row[6 * 64 + static_cast<std::size_t>(square(4, 7))], 1);  // p on e7
    cr_assert_eq(row[11 * 64 + static_cast<std::size_t>(square(4, 8))], 1); // k on e8
}

Test(encoding, onehot_773_adds_side_to_move_and_castling)
{
    const auto base = encode("fen-onehot-768", start);
    const auto row = encode("fen-onehot-773", start);

    cr_assert_eq(row.size(), 773);
    for (std::size_t i = 0; i < 768; i++)
        cr_assert_eq(row[i], base[i], "piece input %zu differs from fen-onehot-768", i);
    cr_assert_eq(countSet(row, 0, 773), 32 + 1 + 4);
    for (std::size_t i = 768; i < 773; i++)
        cr_assert_eq(row[i], 1, "input %zu", i);
}

Test(encoding, onehot_773_has_zeros_for_black_to_move_and_no_castling)
{
    const auto row = encode("fen-onehot-773", kingsBlack);

    cr_assert_eq(countSet(row, 0, 768), 2);
    for (std::size_t i = 768; i < 773; i++)
        cr_assert_eq(row[i], 0, "input %zu", i);
}

Test(encoding, onehot_773_reads_each_castling_right_in_order)
{
    const auto row = encode("fen-onehot-773", "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1");

    cr_assert_eq(row[768], 1);
    cr_assert_eq(row[769], 1); // K
    cr_assert_eq(row[770], 0); // Q
    cr_assert_eq(row[771], 0); // k
    cr_assert_eq(row[772], 1); // q
}

Test(encoding, signed_385_gives_white_plus_one_and_black_minus_one)
{
    const auto row = encode("fen-signed-385", start);

    cr_assert_eq(row.size(), 385);
    cr_assert_eq(countSet(row, 0, 384), 32);
    cr_assert_eq(row[0 * 64 + static_cast<std::size_t>(square(4, 2))], 1);  // P on e2
    cr_assert_eq(row[0 * 64 + static_cast<std::size_t>(square(4, 7))], -1); // p on e7
    cr_assert_eq(row[1 * 64 + static_cast<std::size_t>(square(1, 1))], 1);  // N on b1
    cr_assert_eq(row[1 * 64 + static_cast<std::size_t>(square(1, 8))], -1); // n on b8
    cr_assert_eq(row[4 * 64 + static_cast<std::size_t>(square(3, 1))], 1);  // Q on d1
    cr_assert_eq(row[5 * 64 + static_cast<std::size_t>(square(4, 1))], 1);  // K on e1
    cr_assert_eq(row[5 * 64 + static_cast<std::size_t>(square(4, 8))], -1); // k on e8
    cr_assert_eq(row[384], 1);
    // The two armies cancel plane by plane.
    for (std::size_t plane = 0; plane < 6; plane++) {
        my_torch::Scalar sum = 0;
        for (std::size_t s = 0; s < 64; s++)
            sum += row[plane * 64 + s];
        cr_assert_eq(sum, 0, "plane %zu", plane);
    }
}

Test(encoding, signed_385_side_to_move_is_zero_for_black)
{
    cr_assert_eq(encode("fen-signed-385", kingsBlack)[384], 0);
}

Test(encoding, attacks_901_starts_with_the_773_layout)
{
    const auto base = encode("fen-onehot-773", start);
    const auto row = encode("fen-attacks-901", start);

    cr_assert_eq(row.size(), 901);
    for (std::size_t i = 0; i < 773; i++)
        cr_assert_eq(row[i], base[i], "input %zu differs from fen-onehot-773", i);
}

Test(encoding, attacks_901_in_the_starting_position)
{
    const auto row = encode("fen-attacks-901", start);

    // White attacks b1..g1 (a1 and h1 are covered by nothing), all of rank 2 and all of rank 3:
    // 6 + 8 + 8 = 22 squares; Black mirrors it.
    cr_assert_eq(countSet(row, 773, 837), 22);
    cr_assert_eq(countSet(row, 837, 901), 22);
    for (int file = 0; file < 8; file++) {
        cr_assert(whiteAttacks(row, file, 2));
        cr_assert(whiteAttacks(row, file, 3));
        cr_assert(blackAttacks(row, file, 7));
        cr_assert(blackAttacks(row, file, 6));
        for (int rank = 4; rank <= 8; rank++)
            cr_assert(!whiteAttacks(row, file, rank), "file %d rank %d", file, rank);
        for (int rank = 1; rank <= 5; rank++)
            cr_assert(!blackAttacks(row, file, rank), "file %d rank %d", file, rank);
    }
    cr_assert(!whiteAttacks(row, 0, 1)); // a1
    cr_assert(!whiteAttacks(row, 7, 1)); // h1
    cr_assert(whiteAttacks(row, 1, 1));  // b1, defended by the rook
    cr_assert(!blackAttacks(row, 0, 8)); // a8
    cr_assert(blackAttacks(row, 6, 8));  // g8
}

Test(encoding, attacks_901_a_rook_stops_at_the_first_piece)
{
    // White rook a8 and king a1, Black pawn c8 and king h1.
    const auto row = encode("fen-attacks-901", "R1p5/8/8/8/8/8/8/K6k w - - 0 1");

    cr_assert(whiteAttacks(row, 1, 8));  // b8
    cr_assert(whiteAttacks(row, 2, 8));  // c8: the blocker itself is attacked
    cr_assert(!whiteAttacks(row, 3, 8)); // d8: behind the blocker
    cr_assert(!whiteAttacks(row, 4, 8));
    cr_assert(whiteAttacks(row, 0, 2)); // a2
    cr_assert(whiteAttacks(row, 0, 1)); // a1: the own king is defended
    // The black pawn on c8 attacks b7 and d7, the black king g1, g2 and h2.
    cr_assert(blackAttacks(row, 1, 7));
    cr_assert(blackAttacks(row, 3, 7));
    cr_assert_eq(countSet(row, 837, 901), 2 + 3);
}

Test(encoding, attacks_901_a_knight_in_the_corner_does_not_wrap)
{
    const auto row = encode("fen-attacks-901", "7k/8/8/8/8/8/8/N6K w - - 0 1");

    // Knight a1: b3 and c2. King h1: g1, g2, h2. Nothing else (no wrap to the h file).
    cr_assert_eq(countSet(row, 773, 837), 5);
    cr_assert(whiteAttacks(row, 1, 3));
    cr_assert(whiteAttacks(row, 2, 2));
    cr_assert(whiteAttacks(row, 6, 1));
    cr_assert(whiteAttacks(row, 6, 2));
    cr_assert(whiteAttacks(row, 7, 2));
    // King h8: g8, g7, h7.
    cr_assert_eq(countSet(row, 837, 901), 3);
}

Test(encoding, attacks_901_edge_pawns_attack_one_square_each)
{
    const auto white = encode("fen-attacks-901", "4k3/8/8/8/8/8/P6P/4K3 w - - 0 1");

    // a2 attacks b3, h2 attacks g3; the king on e1 attacks d1, f1, d2, e2, f2.
    cr_assert_eq(countSet(white, 773, 837), 2 + 5);
    cr_assert(whiteAttacks(white, 1, 3));
    cr_assert(whiteAttacks(white, 6, 3));
    cr_assert(!whiteAttacks(white, 0, 3));
    cr_assert(!whiteAttacks(white, 7, 3));

    const auto black = encode("fen-attacks-901", "4k3/p6p/8/8/8/8/8/4K3 b - - 0 1");

    // a7 attacks b6, h7 attacks g6; the king on e8 attacks d8, f8, d7, e7, f7.
    cr_assert_eq(countSet(black, 837, 901), 2 + 5);
    cr_assert(blackAttacks(black, 1, 6));
    cr_assert(blackAttacks(black, 6, 6));
    cr_assert(!blackAttacks(black, 0, 6));
    cr_assert(!blackAttacks(black, 7, 6));
}

Test(encoding, attacks_901_pawns_attack_forward_only)
{
    // White pawn e4 attacks d5 and f5, Black pawn e5 attacks d4 and f4; kings far away.
    const auto row = encode("fen-attacks-901", "k7/8/8/4p3/4P3/8/8/7K w - - 0 1");

    cr_assert(whiteAttacks(row, 3, 5));
    cr_assert(whiteAttacks(row, 5, 5));
    cr_assert(!whiteAttacks(row, 3, 3));
    cr_assert(!whiteAttacks(row, 5, 3));
    cr_assert(blackAttacks(row, 3, 4));
    cr_assert(blackAttacks(row, 5, 4));
    cr_assert(!blackAttacks(row, 3, 6));
    cr_assert(!blackAttacks(row, 5, 6));
}

Test(encoding, attacks_901_a_king_attacks_its_eight_neighbours)
{
    const auto row = encode("fen-attacks-901", "7k/8/8/8/4K3/8/8/8 w - - 0 1");

    cr_assert_eq(countSet(row, 773, 837), 8);
    for (int file = 3; file <= 5; file++)
        for (int rank = 3; rank <= 5; rank++)
            cr_assert_eq(whiteAttacks(row, file, rank), !(file == 4 && rank == 4),
                         "file %d rank %d", file, rank);
}

Test(encoding, attacks_901_a_queen_attacks_rays_up_to_the_first_piece)
{
    // Queen d4 on an empty board attacks 27 squares; the rays to a1 and h8 end on the kings.
    // The white king adds a2 and b1 (b2 is on the queen's diagonal), so 29. Black king: 3.
    const auto row = encode("fen-attacks-901", "7k/8/8/8/3Q4/8/8/K7 w - - 0 1");

    cr_assert_eq(countSet(row, 773, 837), 29);
    cr_assert(whiteAttacks(row, 7, 8)); // h8: the black king is attacked (check)
    cr_assert(whiteAttacks(row, 0, 1));
    cr_assert(whiteAttacks(row, 7, 4));  // h4
    cr_assert(!whiteAttacks(row, 4, 2)); // e2: a knight's jump from d4, not a queen's line
    cr_assert_eq(countSet(row, 837, 901), 3);
}

Test(encoding, encode_rows_stacks_the_encoded_samples)
{
    const std::vector<Sample> samples = {parseLine(start, LabelPolicy::Optional),
                                         parseLine(kingsBlack, LabelPolicy::Optional)};
    const std::string name = "fen-attacks-901";
    const Encoding &encoding = findEncoding(name);
    const my_torch::Matrix rows = encodeRows(encoding, samples, 0, 2);

    cr_assert_eq(rows.rows(), 2);
    cr_assert_eq(rows.cols(), 901);
    const auto first = encode("fen-attacks-901", start);
    const auto second = encode("fen-attacks-901", kingsBlack);
    for (std::size_t c = 0; c < 901; c++) {
        cr_assert_eq(rows(0, c), first[c], "first sample, input %zu", c);
        cr_assert_eq(rows(1, c), second[c], "second sample, input %zu", c);
    }
    cr_assert_eq(encodeRows(encoding, samples, 1, 2)(0, 768), 0);
}
