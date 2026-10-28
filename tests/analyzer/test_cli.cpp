#include <criterion/criterion.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../src/analyzer/Cli.hpp"
#include "../../src/analyzer/Encoding.hpp"
#include "my_torch/my_torch.hpp"

namespace fs = std::filesystem;

namespace {

// Criterion runs tests in parallel processes: every file name carries the process id.
std::string path(const std::string &base)
{
    return "cli_" + std::to_string(getpid()) + "_" + base;
}

void write(const std::string &file, const std::string &text)
{
    std::ofstream(file, std::ios::binary) << text;
}

std::string read(const std::string &file)
{
    std::ifstream in(file, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

struct Result {
    int code;
    std::string out;
};

// Runs the analyzer like main does, capturing standard output.
Result runCli(const std::vector<std::string> &args)
{
    std::ostringstream captured;
    std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
    int code = 0;
    try {
        code = analyzer::run(args);
    } catch (const std::exception &) {
        code = analyzer::EXIT_ERROR;
    }
    std::cout.rdbuf(old);
    return {code, captured.str()};
}

std::vector<std::string> lines(const std::string &text)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);)
        out.push_back(l);
    return out;
}

// Six real positions of the dataset, one per class plus one more Nothing.
const std::vector<std::string> boards = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59",
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 4",
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
    "4r2k/8/8/8/8/8/R7/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
};
const std::vector<std::string> answers = {"Nothing",         "Check White",     "Checkmate White",
                                          "Checkmate White", "Check Black",     "Nothing"};

std::string chessFile(bool withAnswers)
{
    std::string text;
    for (std::size_t i = 0; i < boards.size(); i++)
        text += boards[i] + (withAnswers ? " " + answers[i] : "") + "\n";
    return text;
}

// A small network written from a configuration; labels and training settings as given.
std::string makeNetwork(const std::string &base, const std::string &labels,
                        const std::string &training = "optimizer adam lr=0.01\n"
                                                      "train epochs=300 batch=6 seed=3\n",
                        const std::string &input = "input fen-onehot-768 768\n")
{
    const std::size_t count = std::count(labels.begin(), labels.end(), '\n');
    const std::string conf = path(base + ".conf");
    write(conf, input + "labels " + std::to_string(count) + "\n" + labels + training +
                    "layers 4\ndense 768 16\ntanh\ndense 16 " + std::to_string(count) +
                    "\nsoftmax\n");
    my_torch::Network net = my_torch::Network::fromConfig(conf);
    std::remove(conf.c_str());
    const std::string file = path(base + ".nn");
    net.save(file);
    return file;
}

const std::string fiveLabels =
    "Nothing\nCheck White\nCheck Black\nCheckmate White\nCheckmate Black\n";
const std::string threeLabels = "Nothing\nCheck\nCheckmate\n";

void cleanup()
{
    for (const auto &entry : fs::directory_iterator("."))
        if (entry.path().filename().string().rfind(path(""), 0) == 0)
            fs::remove(entry.path());
}

} // namespace

Test(cli, help_exits_0)
{
    const Result r = runCli({"--help"});
    cr_assert_eq(r.code, 0);
    cr_assert(r.out.find("./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE "
                         "CHESSFILE") != std::string::npos);
}

Test(cli, predict_prints_one_label_per_position_in_file_order)
{
    const std::string net = makeNetwork("order", fiveLabels);
    const std::string chess = path("order.txt");
    write(chess, chessFile(false));
    const Result forward = runCli({"--predict", net, chess});

    std::string reversed;
    for (std::size_t i = boards.size(); i-- > 0;)
        reversed += boards[i] + "\n";
    write(chess, reversed);
    const Result backward = runCli({"--predict", net, chess});
    cleanup();

    cr_assert_eq(forward.code, 0);
    const auto a = lines(forward.out);
    const auto b = lines(backward.out);
    cr_assert_eq(a.size(), boards.size());
    for (std::size_t i = 0; i < a.size(); i++) {
        cr_assert(a[i] == b[a.size() - 1 - i], "line %zu", i);
        cr_assert(fiveLabels.find(a[i] + "\n") != std::string::npos, "'%s'", a[i].c_str());
    }
}

Test(cli, predict_ignores_optional_expected_outputs)
{
    const std::string net = makeNetwork("optional", fiveLabels);
    const std::string with = path("with.txt");
    const std::string without = path("without.txt");
    write(with, chessFile(true));
    write(without, chessFile(false));
    const Result a = runCli({"--predict", net, with});
    const Result b = runCli({"--predict", net, without});
    cleanup();

    cr_assert_eq(a.code, 0);
    cr_assert(a.out == b.out);
}

Test(cli, predict_handles_more_positions_than_one_chunk)
{
    const std::string net = makeNetwork("chunk", fiveLabels);
    const std::string one = path("one.txt");
    const std::string many = path("many.txt");
    write(one, chessFile(false));
    std::string text;
    const std::size_t count = analyzer::PREDICT_CHUNK + 7;
    for (std::size_t i = 0; i < count; i++)
        text += boards[i % boards.size()] + "\n";
    write(many, text);
    const auto single = lines(runCli({"--predict", net, one}).out);
    const auto all = lines(runCli({"--predict", net, many}).out);
    cleanup();

    cr_assert_eq(all.size(), count);
    for (std::size_t i = 0; i < count; i++)
        cr_assert(all[i] == single[i % boards.size()], "line %zu", i);
}

Test(cli, train_then_predict_learns_the_positions)
{
    const std::string net = makeNetwork("learn", fiveLabels);
    const std::string chess = path("learn.txt");
    const std::string boardsOnly = path("boards.txt");
    write(chess, chessFile(true));
    write(boardsOnly, chessFile(false));
    const Result trained = runCli({"--train", net, chess});
    const Result predicted = runCli({"--predict", net, boardsOnly});
    cleanup();

    cr_assert_eq(trained.code, 0);
    cr_assert(trained.out.find("epoch 300/300") != std::string::npos, "%s",
              trained.out.c_str());
    const auto out = lines(predicted.out);
    for (std::size_t i = 0; i < answers.size(); i++)
        cr_assert(out[i] == answers[i], "position %zu: %s", i, out[i].c_str());
}

Test(cli, a_network_without_colors_answers_without_colors)
{
    const std::string net = makeNetwork("nocolor", threeLabels);
    const std::string chess = path("nocolor.txt");
    const std::string boardsOnly = path("nocolor_boards.txt");
    write(chess, chessFile(true));
    write(boardsOnly, chessFile(false));
    cr_assert_eq(runCli({"--train", net, chess}).code, 0);
    const auto out = lines(runCli({"--predict", net, boardsOnly}).out);
    cleanup();

    const std::vector<std::string> expected = {"Nothing",   "Check", "Checkmate",
                                               "Checkmate", "Check", "Nothing"};
    for (std::size_t i = 0; i < expected.size(); i++)
        cr_assert(out[i] == expected[i], "position %zu: %s", i, out[i].c_str());
}

Test(cli, save_keeps_loadfile_and_no_save_overwrites_it)
{
    const std::string net = makeNetwork("save", fiveLabels,
                                        "train epochs=2 batch=6 seed=3\n");
    const std::string chess = path("save.txt");
    const std::string saved = path("saved.nn");
    write(chess, chessFile(true));
    const std::string before = read(net);

    cr_assert_eq(runCli({"--train", "--save", saved, net, chess}).code, 0);
    cr_assert(read(net) == before, "LOADFILE changed");
    cr_assert(fs::exists(saved));
    my_torch::Network::load(saved);

    cr_assert_eq(runCli({"--train", net, chess}).code, 0);
    cr_assert(read(net) != before, "LOADFILE not overwritten");
    my_torch::Network::load(net);
    cleanup();
}

Test(cli, early_stopping_uses_the_validation_fraction)
{
    const std::string net = makeNetwork(
        "valid", fiveLabels, "train epochs=50 batch=6 seed=3 patience=2 validation=0.34\n");
    const std::string chess = path("valid.txt");
    write(chess, chessFile(true));
    const Result r = runCli({"--train", net, chess});
    cleanup();

    cr_assert_eq(r.code, 0);
    cr_assert(r.out.find("validation accuracy") != std::string::npos, "%s", r.out.c_str());
}

Test(cli, every_error_exits_84)
{
    const std::string net = makeNetwork("errors", fiveLabels);
    const std::string twoLabels = makeNetwork("two", "Nothing\nCheck\n");
    const std::string otherEncoding =
        makeNetwork("other", fiveLabels, "", "input board-2d 768\n");
    const std::string patienceOnly =
        makeNetwork("patience", fiveLabels, "train epochs=2 batch=6 seed=3 patience=2\n");
    const std::string chess = path("errors.txt");
    const std::string unlabeled = path("unlabeled.txt");
    const std::string badFen = path("badfen.txt");
    const std::string corrupted = path("corrupted.nn");
    write(chess, chessFile(true));
    write(unlabeled, chessFile(false));
    write(badFen, boards[0] + "\n8/8/8 w - - 0 1\n");
    write(corrupted, read(net).substr(0, 500));

    const std::vector<std::vector<std::string>> cases = {
        {},
        {net, chess},
        {"--predict", "--train", net, chess},
        {"--predict", "--save", "x.nn", net, chess},
        {"--train", "--save"},
        {"--train", "--save", "a.nn", "--save", "b.nn", net, chess},
        {"--predict", net},
        {"--predict", net, chess, "extra"},
        {"--predict", "--verbose", net, chess},
        {"--predict", "missing.nn", chess},
        {"--predict", net, "missing.txt"},
        {"--predict", corrupted, chess},
        {"--predict", otherEncoding, chess},
        {"--train", net, unlabeled},
        {"--train", twoLabels, chess},
        {"--train", patienceOnly, chess},
    };
    for (const auto &args : cases) {
        std::string shown;
        for (const auto &a : args)
            shown += a + " ";
        cr_assert_eq(runCli(args).code, 84, "%s", shown.c_str());
    }
    // A malformed line: nothing is printed, not even the predictions of the valid lines.
    const Result r = runCli({"--predict", net, badFen});
    cleanup();
    cr_assert_eq(r.code, 84);
    cr_assert(r.out.empty(), "printed: %s", r.out.c_str());
}

Test(cli, label_index_matches_exactly_or_by_first_word)
{
    const std::vector<std::string> labels = {"Nothing", "Check", "Checkmate"};
    cr_assert_eq(analyzer::labelIndex("Checkmate", labels), 2);
    cr_assert_eq(analyzer::labelIndex("Check White", labels), 1);
    cr_assert_eq(analyzer::labelIndex("Checkmate Black", labels), 2);
    const std::vector<std::string> colors = {"Check White", "Check"};
    cr_assert_eq(analyzer::labelIndex("Check White", colors), 0); // exact match wins
    cr_assert_throw(analyzer::labelIndex("Stalemate", labels), my_torch::Error);
}

Test(encoding, one_hot_768_of_the_starting_position)
{
    const std::string name = "fen-onehot-768";
    const analyzer::Encoding &e = analyzer::findEncoding(name);
    std::vector<my_torch::Scalar> row(e.width(), 0);
    e.encode(analyzer::parseFen(boards[0]), row.data());

    cr_assert_eq(e.width(), 768);
    std::size_t ones = 0;
    for (const auto v : row)
        ones += v == 1;
    cr_assert_eq(ones, 32);
    // Plane 0 is the white pawns: a2 is square 48. Plane 11 is the black king: e8 is square 4.
    cr_assert_eq(row[0 * 64 + 48], 1);
    cr_assert_eq(row[11 * 64 + 4], 1);
    cr_assert_eq(row[0 * 64 + 40], 0); // a3 is empty
    cr_assert_throw(analyzer::findEncoding("board-2d"), my_torch::Error);
}

Test(cli, validation_tail_lines_are_never_trained)
{
    // Two files with the same six training lines and two different last lines: with
    // validation_tail=2 the trained networks must be byte-identical.
    const std::string training = "train epochs=20 batch=3 seed=3 validation_tail=2\n";
    const std::string netA = makeNetwork("tail_a", fiveLabels, training);
    const std::string netB = makeNetwork("tail_b", fiveLabels, training);
    const std::string fileA = path("tail_a.txt");
    const std::string fileB = path("tail_b.txt");
    write(fileA, chessFile(true) + boards[0] + " Nothing\n" + boards[1] + " Check White\n");
    write(fileB, chessFile(true) + boards[2] + " Checkmate White\n" + boards[4] + " Check Black\n");
    const Result a = runCli({"--train", netA, fileA});
    const Result b = runCli({"--train", netB, fileB});
    const bool same = read(netA) == read(netB);
    cleanup();

    cr_assert_eq(a.code, 0);
    cr_assert_eq(b.code, 0);
    cr_assert(a.out.find("validation accuracy") != std::string::npos, "%s", a.out.c_str());
    cr_assert(same, "the last lines changed the training");
}

Test(cli, validation_tail_errors)
{
    const std::string tooLong =
        makeNetwork("tail_long", fiveLabels, "train epochs=1 batch=6 seed=3 validation_tail=6\n");
    const std::string chess = path("tail_err.txt");
    write(chess, chessFile(true));
    const int code = runCli({"--train", tooLong, chess}).code;
    cleanup();
    cr_assert_eq(code, 84); // the validation part leaves no training position

    const std::string conf = path("both.conf");
    write(conf, "input fen-onehot-768 768\nlabels 1\nNothing\n"
                "train validation=0.1 validation_tail=5\nlayers 2\ndense 768 1\nsoftmax\n");
    cr_assert_throw(my_torch::Network::fromConfig(conf), my_torch::Error);
    std::remove(conf.c_str());
}

Test(cli, train_with_a_checkpoint_resumes_from_it)
{
    const std::string training = "optimizer adam lr=0.01\ntrain epochs=4 batch=6 seed=3\n";
    const std::string net = makeNetwork("ckpt", fiveLabels, training);
    const std::string chess = path("ckpt.txt");
    const std::string checkpoint = path("ckpt.ckpt");
    const std::string once = path("once.nn");
    const std::string twice = path("twice.nn");
    write(chess, chessFile(true));

    cr_assert_eq(runCli({"--train", "--save", once, "--checkpoint", checkpoint, net, chess}).code,
                 0);
    cr_assert(fs::exists(checkpoint));
    // Run again: the 4 epochs are in the checkpoint, so nothing is trained and the network is
    // the same; the epochs are still listed.
    const Result again =
        runCli({"--train", "--save", twice, "--checkpoint", checkpoint, net, chess});
    const bool same = read(once) == read(twice);
    cleanup();

    cr_assert_eq(again.code, 0);
    cr_assert(again.out.find("epoch 4/4") != std::string::npos, "%s", again.out.c_str());
    cr_assert(same, "the resumed network differs");
}

Test(cli, checkpoint_only_works_with_train)
{
    try {
        analyzer::run({"--predict", "--checkpoint", "x.ckpt", "a.nn", "b.txt"});
        cr_assert_fail("--checkpoint was accepted with --predict");
    } catch (const my_torch::Error &e) {
        cr_assert(std::string(e.what()).find("--checkpoint only works with --train") !=
                      std::string::npos,
                  "%s", e.what());
    }
    cr_assert_eq(runCli({"--train", "--checkpoint"}).code, analyzer::EXIT_ERROR);
}
