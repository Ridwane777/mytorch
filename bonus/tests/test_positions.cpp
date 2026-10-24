#include <criterion/criterion.h>

#include "Positions.hpp"
#include "my_torch/Error.hpp"

using namespace positions;

Test(positions, fen_round_trip)
{
    for (const char *fen : {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                            "8/k7/8/8/8/6q1/8/7K w - - 0 1", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 3 42"})
        cr_assert(toFen(analyzer::parseFen(fen)) == fen, "%s", fen);
}

Test(positions, generated_positions_are_legal_and_labeled_by_the_oracle)
{
    Report report;
    const auto all = generate(400, 11, 0.25, &report);
    std::size_t stalemates = 0;

    cr_assert_eq(all.size(), 400);
    for (const Labeled &l : all) {
        // Parsable FEN with one king per color, as the analyzer requires.
        const analyzer::Position p = analyzer::parseFen(toFen(l.position));
        cr_assert(oracle::classify(p) == l.verdict);
        cr_assert(l.verdict != oracle::Verdict::Illegal);
        for (int file = 0; file < 8; file++) {
            cr_assert_neq(std::toupper(p.board[analyzer::square(file, 1)]), 'P');
            cr_assert_neq(std::toupper(p.board[analyzer::square(file, 8)]), 'P');
        }
        stalemates += l.verdict == oracle::Verdict::Stalemate;
    }
    cr_assert_eq(stalemates, 100);
    cr_assert(report.legal <= report.drawn && report.stalemates >= 100);
}

Test(positions, same_seed_same_positions)
{
    const auto a = generate(50, 3, 0.5);
    const auto b = generate(50, 3, 0.5);
    const auto c = generate(50, 4, 0.5);
    bool differ = false;

    for (std::size_t i = 0; i < a.size(); i++) {
        cr_assert(toFen(a[i].position) == toFen(b[i].position));
        differ = differ || toFen(a[i].position) != toFen(c[i].position);
    }
    cr_assert(differ);
}

Test(positions, labels_are_not_sorted)
{
    const auto all = generate(100, 5, 0.5);
    std::size_t changes = 0;
    for (std::size_t i = 1; i < all.size(); i++)
        changes += (all[i].verdict == oracle::Verdict::Stalemate) !=
                   (all[i - 1].verdict == oracle::Verdict::Stalemate);
    cr_assert_gt(changes, 20);
}

Test(positions, rejects_a_bad_share)
{
    cr_assert_throw(generate(10, 1, 1.5), my_torch::Error);
    cr_assert_throw(generate(10, 1, -0.1), my_torch::Error);
    cr_assert_eq(generate(10, 1, 0).size(), 10);
}
