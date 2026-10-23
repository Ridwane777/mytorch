#include <criterion/criterion.h>

#include "Oracle.hpp"

using namespace oracle;
using analyzer::parseFen;
using analyzer::square;

namespace {

constexpr const char *start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Position de test classique sans roque possible, avec prises en passant et promotions.
constexpr const char *endgame = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

Verdict verdictOf(const char *fen)
{
    return classify(parseFen(fen));
}

} // namespace

Test(attacks, rook_attacks_along_an_open_rank)
{
    Position p = parseFen("8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59");

    cr_assert(isAttacked(p, square(3, 6), Color::White));
    cr_assert(inCheck(p, Color::Black));
    cr_assert(!inCheck(p, Color::White));
}

Test(attacks, a_piece_in_between_blocks_the_rook)
{
    cr_assert(!inCheck(parseFen("8/8/R1pk4/8/8/5K2/8/8 b - - 0 1"), Color::Black));
}

Test(attacks, knight_jumps_over_pieces)
{
    // Cavalier d6 : il attaque e8 par-dessus les pions d7 et e7.
    cr_assert(inCheck(parseFen("4k3/2pppp2/3N4/8/8/8/8/4K3 b - - 0 1"), Color::Black));
    cr_assert(!inCheck(parseFen("4k3/2pppp2/4N3/8/8/8/8/4K3 b - - 0 1"), Color::Black));
}

Test(attacks, pawns_attack_forward_only)
{
    // Un pion blanc en d7 attaque e8 ; un pion blanc en d2 n'attaque pas e1, derrière lui.
    cr_assert(inCheck(parseFen("4k3/3P4/8/8/8/8/8/4K3 b - - 0 1"), Color::Black));
    cr_assert(!inCheck(parseFen("8/8/8/8/8/8/3P4/4k2K b - - 0 1"), Color::Black));
    cr_assert(inCheck(parseFen("8/8/8/8/8/8/3p4/4K2k w - - 0 1"), Color::White));
}

Test(verdict, known_positions)
{
    cr_assert(verdictOf(start) == Verdict::Nothing);
    cr_assert(verdictOf("8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59") == Verdict::CheckWhite);
    // Mat du berger.
    cr_assert(verdictOf("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4") ==
              Verdict::CheckmateWhite);
    // Mat du couloir : le roi est enfermé par ses propres pions.
    cr_assert(verdictOf("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1") == Verdict::CheckmateWhite);
    // Mat à l'étouffée par un cavalier.
    cr_assert(verdictOf("6rk/5Npp/8/8/8/8/8/6K1 b - - 0 1") == Verdict::CheckmateWhite);
    // Échec noir que le roi blanc peut fuir.
    cr_assert(verdictOf("4r2k/8/8/8/8/8/R7/4K3 w - - 0 1") == Verdict::CheckBlack);
    // Mat noir : dame protégée par le roi.
    cr_assert(verdictOf("8/8/8/8/8/5k2/6q1/7K w - - 0 1") == Verdict::CheckmateBlack);
    // Pat : le roi noir n'est pas attaqué mais n'a aucune case.
    cr_assert(verdictOf("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") == Verdict::Stalemate);
    // Les Blancs ont le trait alors que le roi noir est déjà en échec.
    cr_assert(verdictOf("R3k3/8/8/8/8/8/8/4K3 w - - 0 1") == Verdict::Illegal);
}

Test(verdict, blocking_turns_a_mate_into_a_check)
{
    // Mat du couloir, mais la tour noire c2 peut s'interposer en c8.
    cr_assert(verdictOf("R5k1/5ppp/8/8/8/8/2r5/6K1 b - - 0 1") == Verdict::CheckWhite);
    // Le fou noir f7 peut s'interposer en e8.
    cr_assert(verdictOf("R5k1/5bpp/8/8/8/8/8/6K1 b - - 0 1") == Verdict::CheckWhite);
}

Test(verdict, a_pinned_piece_cannot_block)
{
    // Même position, mais le fou blanc c4 cloue le fou f7 sur la diagonale du roi g8 : s'il va en
    // e8, son roi est attaqué. Aucun coup ne sauve, c'est mat.
    cr_assert(verdictOf("R5k1/5bpp/8/8/2B5/8/8/6K1 b - - 0 1") == Verdict::CheckmateWhite);
}

Test(moves, en_passant_is_generated_and_removes_the_pawn)
{
    Position p = parseFen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    bool found = false;

    for (const Move &m : legalMoves(p))
        if (m.enPassant) {
            found = true;
            Position next = play(p, m);
            cr_assert_eq(next.board[square(3, 6)], 'P');
            cr_assert_eq(next.board[square(3, 5)], '.');
            cr_assert_eq(next.board[square(4, 5)], '.');
        }
    cr_assert(found);
}

Test(moves, promotion_generates_four_moves)
{
    cr_assert_eq(legalMoves(parseFen("8/P6k/8/8/8/8/8/K7 w - - 0 1")).size(), 4 + 3);
}

Test(perft, starting_position)
{
    Position p = parseFen(start);

    cr_assert_eq(perft(p, 1), 20);
    cr_assert_eq(perft(p, 2), 400);
    cr_assert_eq(perft(p, 3), 8902);
    cr_assert_eq(perft(p, 4), 197281);
}

Test(perft, endgame_with_en_passant_and_promotions)
{
    Position p = parseFen(endgame);

    cr_assert_eq(perft(p, 1), 14);
    cr_assert_eq(perft(p, 2), 191);
    cr_assert_eq(perft(p, 3), 2812);
    cr_assert_eq(perft(p, 4), 43238);
    cr_assert_eq(perft(p, 5), 674624);
}

Test(labels, verdicts_convert_to_subject_labels)
{
    cr_assert(toLabel(Verdict::CheckmateBlack) == analyzer::Label::CheckmateBlack);
    cr_assert(!toLabel(Verdict::Stalemate).has_value());
    cr_assert(!toLabel(Verdict::Illegal).has_value());
}
