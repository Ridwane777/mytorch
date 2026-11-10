#include <criterion/criterion.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: one file name per process.
const std::string file = "serialization_test_" + std::to_string(getpid()) + ".nn";

// One layer of every type, with non-default settings everywhere.
Network everyLayer()
{
    Network net;
    net.add(std::make_unique<Dense>(3, 4, 1, Regularization{0.01, 0.02}));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dense>(4, 4, 2));
    net.add(std::make_unique<LeakyReLU>(0.2));
    net.add(std::make_unique<Dropout>(0.25, 9));
    net.add(std::make_unique<Dense>(4, 4, 3));
    net.add(std::make_unique<Sigmoid>());
    net.add(std::make_unique<Dense>(4, 4, 4));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(4, 2, 5));
    net.add(std::make_unique<Softmax>());
    ModelInfo &info = net.info();
    info.encoding = "test-encoding";
    info.inputs = 3;
    info.labels = {"Check White", "Nothing"};
    info.training.loss = {"cross_entropy", {{"smoothing", "0.1"}, {"weights", "1,2"}}};
    info.training.optimizer = {"adam", {{"lr", "0.01"}, {"beta1", "0.8"}}};
    info.training.scheduler = {"cosine", {{"base", "0.01"}, {"epochs", "5"}, {"warmup", "1"}}};
    info.training.config.epochs = 7;
    info.training.config.batchSize = 3;
    info.training.config.seed = 99;
    info.training.config.shuffle = false;
    info.training.config.patience = 2;
    info.training.config.minDelta = 0.001;
    info.training.config.restoreBest = false;
    info.training.config.validationTail = 17;
    return net;
}

std::string readText(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

void writeText(const std::string &path, const std::string &text)
{
    std::ofstream(path, std::ios::binary) << text;
}

std::string loadError(const std::string &text)
{
    writeText(file, text);
    std::string message;
    try {
        Network::load(file);
    } catch (const Error &error) {
        message = error.what();
    }
    std::remove(file.c_str());
    return message;
}

// A valid file of every layer type, in the text format (ADR-0012).
std::string savedText()
{
    everyLayer().saveText(file);
    const std::string text = readText(file);
    std::remove(file.c_str());
    return text;
}

std::string replace(std::string text, const std::string &from, const std::string &to)
{
    const std::size_t at = text.find(from);
    cr_assert_neq(at, std::string::npos, "'%s' not in the file", from.c_str());
    return text.replace(at, from.size(), to);
}

// A small hand-written file: one Dense 2 -> 1 with weights 2, 3 and bias 1.
std::string handWritten(const std::string &layers = "layers 1\ndense 2 1\n2\n3\n1\n",
                        const std::string &labels = "labels 1\nout\n")
{
    return "my_torch 1\ninput raw 2\n" + labels +
           "loss mse\noptimizer sgd lr=0.1\nscheduler none\n"
           "train epochs=1 batch=1 seed=1 shuffle=0 patience=0 min_delta=0 restore_best=1\n" +
           layers + "end\n";
}

} // namespace

Test(serialization, round_trip_gives_bit_identical_predictions)
{
    Network original = everyLayer();
    const Matrix x = gradient_check::random(5, 3, 7);

    original.save(file);
    Network loaded = Network::load(file);
    std::remove(file.c_str());
    const Matrix a = original.predict(x);
    const Matrix b = loaded.predict(x);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i], "value %zu differs", i);
    cr_assert_eq(loaded.layers().size(), original.layers().size());
    for (std::size_t i = 0; i < loaded.layers().size(); i++)
        cr_assert(loaded.layers()[i]->type() == original.layers()[i]->type());
}

Test(serialization, round_trip_keeps_settings_and_model_info)
{
    Network original = everyLayer();
    original.save(file);
    Network loaded = Network::load(file);
    std::remove(file.c_str());

    const auto *dense = dynamic_cast<const Dense *>(loaded.layers()[0].get());
    cr_assert_eq(dense->regularization().l1, 0.01);
    cr_assert_eq(dense->regularization().l2, 0.02);
    cr_assert_eq(dynamic_cast<const LeakyReLU *>(loaded.layers()[3].get())->slope(), 0.2);
    const auto *dropout = dynamic_cast<const Dropout *>(loaded.layers()[4].get());
    cr_assert_eq(dropout->rate(), 0.25);
    cr_assert_eq(dropout->seed(), 9u);

    const ModelInfo &a = original.info();
    const ModelInfo &b = loaded.info();
    cr_assert(b.encoding == a.encoding);
    cr_assert_eq(b.inputs, a.inputs);
    cr_assert(b.labels == a.labels);
    cr_assert(b.training.loss.type == "cross_entropy" &&
              b.training.loss.params == a.training.loss.params);
    cr_assert(b.training.optimizer.params == a.training.optimizer.params);
    cr_assert(b.training.scheduler.type == "cosine" &&
              b.training.scheduler.params == a.training.scheduler.params);
    const TrainConfig &c = b.training.config;
    cr_assert(c.epochs == 7 && c.batchSize == 3 && c.seed == 99 && !c.shuffle && c.patience == 2 &&
              c.minDelta == 0.001 && !c.restoreBest && c.validationTail == 17);
}

Test(serialization, saving_a_loaded_network_gives_the_same_file)
{
    const std::string first = savedText();
    writeText(file, first);
    Network::load(file).saveText(file);
    const std::string second = readText(file);
    std::remove(file.c_str());
    cr_assert(first == second);
}

Test(serialization, trained_network_round_trip)
{
    Matrix x(4, 2);
    x(1, 1) = 1;
    x(2, 0) = 1;
    x(3, 0) = 1;
    x(3, 1) = 1;
    Matrix t(4, 2);
    t(0, 0) = 1;
    t(1, 1) = 1;
    t(2, 1) = 1;
    t(3, 0) = 1;
    Network net;
    net.add(std::make_unique<Dense>(2, 8, 1));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(8, 2, 2));
    net.add(std::make_unique<Softmax>());
    net.info() = {"xor", 2, {"0", "1"}, {}};
    CrossEntropy loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 500;
    config.batchSize = 4;
    net.fit({x, t}, nullptr, loss, adam, config);

    net.save(file);
    Network loaded = Network::load(file);
    std::remove(file.c_str());
    cr_assert_eq(report(loaded.predict(x), t).accuracy, 1);
    const Matrix a = net.predict(x);
    const Matrix b = loaded.predict(x);
    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i]);
}

Test(serialization, training_continues_identically_after_a_reload)
{
    // Plain SGD keeps no state and the order is fixed, so 10 + 10 epochs through a file must give
    // the same weights as 20 epochs in one go. (Adam's moments are not saved: #35.)
    Matrix x = gradient_check::random(12, 3, 1);
    Matrix t(12, 2);
    for (std::size_t r = 0; r < 12; r++)
        t(r, r % 2) = 1;
    TrainConfig config;
    config.batchSize = 4;
    config.shuffle = false;
    CrossEntropy loss;
    const auto build = [] {
        Network net;
        net.add(std::make_unique<Dense>(3, 5, 1));
        net.add(std::make_unique<Tanh>());
        net.add(std::make_unique<Dense>(5, 2, 2));
        net.add(std::make_unique<Softmax>());
        net.info() = {"raw", 3, {"a", "b"}, {}};
        return net;
    };

    Network direct = build();
    SGD sgd(0.1);
    config.epochs = 20;
    direct.fit({x, t}, nullptr, loss, sgd, config);

    Network split = build();
    SGD first(0.1);
    config.epochs = 10;
    split.fit({x, t}, nullptr, loss, first, config);
    split.save(file);
    Network reloaded = Network::load(file);
    std::remove(file.c_str());
    SGD second(0.1);
    reloaded.fit({x, t}, nullptr, loss, second, config);

    const auto a = direct.parameters();
    const auto b = reloaded.parameters();
    for (std::size_t p = 0; p < a.size(); p++)
        for (std::size_t i = 0; i < a[p].value->size(); i++)
            cr_assert_eq(a[p].value->data()[i], b[p].value->data()[i]);
}

Test(serialization, loads_a_hand_written_file)
{
    writeText(file, handWritten());
    Network net = Network::load(file);
    std::remove(file.c_str());
    Matrix x(1, 2, 1);

    cr_assert_float_eq(net.predict(x)(0, 0), 6, 1e-12); // 2·1 + 3·1 + 1
    cr_assert(net.info().labels == std::vector<std::string>{"out"});
    cr_assert(net.info().training.optimizer.type == "sgd");
}

Test(serialization, rejects_malformed_files)
{
    const std::string good = savedText();
    struct Case {
        const char *what;
        std::string text;
    };
    const std::vector<Case> cases = {
        {"not a my_torch network file", replace(good, "my_torch 1", "hello 1")},
        {"format version 2", replace(good, "my_torch 1", "my_torch 2")},
        {"expected 'labels'", replace(good, "labels 2", "classes 2")},
        {"unknown layer type 'conv9d'", replace(good, "\nrelu\n", "\nconv9d\n")},
        {"unexpected end of file", good.substr(0, good.find("softmax"))},
        {"unexpected data after 'end'", good + "garbage\n"},
        {"checksum mismatch", replace(good, "l2=0.02", "l2=0.03")},
        {"train: unknown setting 'speed'",
         replace(good, "train epochs=7", "train speed=1 epochs=7")},
        {"dense: unexpected 'l3=1'", replace(good, "l2=0.02", "l2=0.02 l3=1")},
        {"dropout: rate must be in [0, 1[", replace(good, "rate=0.25", "rate=1.5")},
        {"regularization coefficients must be non-negative", replace(good, "l2=0.02", "l2=-1")},
    };
    for (const Case &c : cases) {
        const std::string message = loadError(c.text);
        cr_assert(message.find(c.what) != std::string::npos, "%s -> %s", c.what, message.c_str());
        cr_assert(message.find(file + ":") == 0, "no file position: %s", message.c_str());
    }
}

Test(serialization, checksum_catches_one_changed_weight_digit)
{
    std::string text = savedText();
    // First weight of the first dense layer: change its last digit, the file still parses.
    const std::size_t line = text.find('\n', text.find("dense 3 4")) + 1;
    std::size_t digit = text.find(' ', line) - 1;
    text[digit] = text[digit] == '1' ? '2' : '1';
    const std::string message = loadError(text);
    cr_assert(message.find("checksum mismatch") != std::string::npos, "%s", message.c_str());
}

Test(serialization, init_key_is_refused_in_a_network_file)
{
    // `init` belongs to configurations: a network file stores its weights.
    const std::string message =
        loadError(handWritten("layers 1\ndense 2 1 init=he_normal\n2\n3\n1\n"));

    cr_assert(message.find("dense: unexpected 'init=he_normal'") != std::string::npos, "%s",
              message.c_str());
}

Test(serialization, rejects_bad_numbers)
{
    for (const char *bad : {"abc", "nan", "inf", "1e999"}) {
        const std::string text =
            handWritten(std::string("layers 1\ndense 2 1\n") + bad + "\n3\n1\n");
        const std::string message = loadError(text);
        cr_assert(message.find("is not a finite number") != std::string::npos, "%s: %s", bad,
                  message.c_str());
    }
}

Test(serialization, rejects_truncated_weights)
{
    const std::string message = loadError(handWritten("layers 1\ndense 2 2\n1 2\n3\n0 0\n"));
    cr_assert(message.find("expected 2 numbers, got 1") != std::string::npos, "%s",
              message.c_str());
}

Test(serialization, rejects_inconsistent_dimensions)
{
    // Input width 2 but the dense layer takes 3.
    std::string message = loadError(handWritten("layers 1\ndense 3 1\n1\n1\n1\n0\n"));
    cr_assert(message.find("expects 3 inputs but receives 2") != std::string::npos, "%s",
              message.c_str());
    // Two outputs for one label.
    message = loadError(handWritten("layers 1\ndense 2 2\n1 1\n1 1\n0 0\n"));
    cr_assert(message.find("2 outputs for 1 labels") != std::string::npos, "%s", message.c_str());
}

Test(serialization, save_refuses_an_incomplete_model)
{
    Network net;
    net.add(std::make_unique<Dense>(2, 1));
    cr_assert_throw(net.save(file), Error); // no encoding, inputs or labels
    net.info() = {"raw", 2, {"out"}, {}};
    cr_assert_throw(net.save("/nonexistent/dir/net.nn"), Error);
    cr_assert_throw(Network::load("/nonexistent/net.nn"), Error);
}

// ---- binary format (ADR-0042) ------------------------------------------------------------------

namespace {

std::string savedBinary(const Network &net)
{
    net.save(file);
    const std::string bytes = readText(file);
    std::remove(file.c_str());
    return bytes;
}

std::uint64_t little(const std::string &bytes, std::size_t at, int count)
{
    std::uint64_t value = 0;
    for (int i = 0; i < count; i++)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[at + i])) << (8 * i);
    return value;
}

} // namespace

Test(binary_format, save_writes_binary_with_no_text_record)
{
    const std::string bytes = savedBinary(everyLayer());

    cr_assert(bytes.compare(0, 8, std::string("MYTORCH\0", 8)) == 0, "no signature");
    cr_assert_eq(little(bytes, 8, 4), 2, "version");
    cr_assert_eq(little(bytes, 12, 8), bytes.size() - 28, "payload length");
    // Keywords are one-byte codes and numbers binary: no record survives as text.
    for (const char *text : {"my_torch 1", "dense", "relu", "epochs", "checksum", "beta1"})
        cr_assert(bytes.find(text) == std::string::npos, "'%s' written as text", text);
}

Test(binary_format, round_trip_keeps_every_bit_and_setting)
{
    const Network original = everyLayer();
    original.save(file);
    Network::load(file).saveText(file);
    const std::string again = readText(file);
    std::remove(file.c_str());

    cr_assert(again == savedText(), "the text of the reloaded network differs");
}

Test(binary_format, is_smaller_than_the_text_format)
{
    Network net;
    net.add(std::make_unique<Dense>(64, 64, 4));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "raw";
    net.info().inputs = 64;
    net.info().labels.resize(64, "x");
    net.saveText(file);
    const std::size_t text = readText(file).size();

    cr_assert_lt(savedBinary(net).size() * 2, text, "binary %zu, text %zu", savedBinary(net).size(),
                 text);
}

Test(binary_format, labels_with_any_spacing_come_back_unchanged)
{
    Network net;
    net.add(std::make_unique<Dense>(2, 3, 1));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "my-encoding";
    net.info().inputs = 2;
    net.info().labels = {"Check  White", " leading", "\xc3\xa9 unicode"};
    net.save(file);
    const Network loaded = Network::load(file);
    std::remove(file.c_str());

    cr_assert(loaded.info().labels == net.info().labels);
    cr_assert(loaded.info().encoding == "my-encoding");
}

Test(binary_format, rejects_corrupted_truncated_and_unknown_files)
{
    const std::string good = savedBinary(everyLayer());
    std::string flipped = good;
    flipped[40] = static_cast<char>(flipped[40] ^ 0x01);
    std::string newer = good;
    newer[8] = 3;
    struct Case {
        const char *what;
        std::string bytes;
    };
    const std::vector<Case> cases = {
        {"checksum mismatch", flipped},
        {"truncated", good.substr(0, good.size() - 5)},
        {"truncated", good.substr(0, 12)},
        {"truncated or has extra data", good + "x"},
        {"binary format version 3 is not supported", newer},
    };
    for (const Case &c : cases) {
        const std::string message = loadError(c.bytes);
        cr_assert(message.find(c.what) != std::string::npos, "%s -> %s", c.what, message.c_str());
        cr_assert(message.find(file + ":") == 0, "no file name: %s", message.c_str());
    }
}

Test(binary_format, text_files_written_before_still_load)
{
    writeText(file, savedText());
    Network fromText = Network::load(file);
    fromText.save(file);
    Network fromBinary = Network::load(file);
    std::remove(file.c_str());
    const Matrix x = gradient_check::random(4, 3, 11);
    const Matrix a = fromText.predict(x);
    const Matrix b = fromBinary.predict(x);

    for (std::size_t i = 0; i < a.size(); i++)
        cr_assert_eq(a.data()[i], b.data()[i], "value %zu differs", i);
}
