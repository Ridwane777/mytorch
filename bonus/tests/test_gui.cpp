#include <criterion/criterion.h>

#include <string>

#include "Analysis.hpp"
#include "Encoding.hpp"
#include "Fen.hpp"
#include "Oracle.hpp"
#include "Saliency.hpp"
#include "my_torch/Error.hpp"

namespace {

// The pushed network; the tests run from bonus/.
constexpr const char *network = "../my_torch_network_final.nn";
constexpr const char *start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

bool contains(const std::string &text, const std::string &part)
{
    return text.find(part) != std::string::npos;
}

} // namespace

Test(gui, url_decoding)
{
    cr_assert_eq(gui::urlDecode("8%2F8+w%20-"), "8/8 w -");
    cr_assert_eq(gui::urlDecode("abc"), "abc");
    cr_assert_throw(gui::urlDecode("%2"), my_torch::Error);
    cr_assert_throw(gui::urlDecode("%zz"), my_torch::Error);
}

Test(gui, query_value)
{
    cr_assert_eq(gui::queryValue("a=1&fen=8%2F8&b=2", "fen"), "8/8");
    cr_assert_eq(gui::queryValue("a=1", "fen"), "");
    cr_assert_eq(gui::queryValue("", "fen"), "");
    cr_assert_eq(gui::queryValue("fenx=1&fen=2", "fen"), "2");
}

Test(gui, json_escaping)
{
    cr_assert_eq(gui::jsonEscape("a\"b\\c"), "a\\\"b\\\\c");
    cr_assert_eq(gui::jsonEscape("\n"), "\\u000a");
    cr_assert_eq(gui::errorJson("bad \"fen\""), "{\"error\":\"bad \\\"fen\\\"\"}");
}

Test(gui, analysis_of_the_start_position)
{
    gui::Analysis analysis(network);
    const std::string json = analysis.analyze(start);

    cr_assert(contains(json, std::string("\"fen\":\"") + start + "\""));
    cr_assert(contains(json, "\"prediction\":\"Nothing\""));
    cr_assert(contains(json, "\"oracle\":\"Nothing\""));
    cr_assert(contains(json, "{\"label\":\"Checkmate Black\",\"p\":"));
}

Test(gui, oracle_verdict_matches_the_oracle)
{
    // Fool's mate: White is mated.
    const char *mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
    gui::Analysis analysis(network);
    const std::string expected = oracle::verdictName(oracle::classify(analyzer::parseFen(mate)));

    cr_assert(contains(analysis.analyze(mate), "\"oracle\":\"" + expected + "\""));
}

Test(gui, invalid_fen_throws)
{
    gui::Analysis analysis(network);

    cr_assert_throw(analysis.analyze("not a fen"), my_torch::Error);
    cr_assert_throw(analysis.analyze(""), my_torch::Error);
}

Test(gui, saliency_points_at_the_mating_queen)
{
    // Fool's mate: the queen on h4 mates the king on e1. Removing the queen must cost the most
    // probability of the network's answer, and the JSON carries the 64 squares of both maps.
    const char *mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
    my_torch::Network net = my_torch::Network::load(network);
    const analyzer::Encoding &encoding = analyzer::findEncoding(net.info().encoding);
    const gui::SquareSaliency s = gui::saliency(net, encoding, analyzer::parseFen(mate));
    const auto h4 = static_cast<std::size_t>(analyzer::square(7, 4));

    std::size_t top = 0;
    for (std::size_t i = 1; i < 64; i++)
        if (s.occlusion[i] > s.occlusion[top])
            top = i;
    cr_assert_eq(top, h4, "most important square %zu", top);
    cr_assert_eq(s.occlusion[static_cast<std::size_t>(analyzer::square(4, 4))], 0, "an empty square has no occlusion");
    gui::Analysis analysis(network);
    const std::string json = analysis.analyze(mate);
    cr_assert(contains(json, "\"gradient\":[") && contains(json, "\"occlusion\":["));
}
