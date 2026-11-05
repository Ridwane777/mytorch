#include <criterion/criterion.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// Criterion runs tests in parallel processes: one file name per process.
const std::string file = "ensemble_test_" + std::to_string(getpid()) + ".nn";

// 3 inputs, a dense layer drawn from seed, softmax over the labels.
Network member(std::uint64_t seed, std::vector<std::string> labels = {"a", "b", "c"})
{
    Network net;
    net.add(std::make_unique<Dense>(3, labels.size(), seed));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "test";
    net.info().inputs = 3;
    net.info().labels = std::move(labels);
    return net;
}

// Reads members × 3 probabilities, gives 3.
Network combiner(std::size_t members)
{
    Network net;
    net.add(std::make_unique<Dense>(members * 3, 3, 7));
    net.add(std::make_unique<Softmax>());
    net.info().encoding = "probabilities";
    net.info().inputs = members * 3;
    net.info().labels = {"a", "b", "c"};
    return net;
}

Matrix inputs()
{
    Matrix x(4, 3);
    const double values[] = {1, -2, 0.5, 0, 3, -1, -1, -1, 2, 0.25, 0.5, -0.75};
    for (std::size_t i = 0; i < 12; i++)
        x.data()[i] = values[i];
    return x;
}

std::string readBytes(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

void writeBytes(const std::string &path, const std::string &bytes)
{
    std::ofstream(path, std::ios::binary) << bytes;
}

} // namespace

Test(ensemble, average_is_the_weighted_mean)
{
    Ensemble e(Combination::Average);
    e.add(member(1), 1);
    e.add(member(2), 3);
    const Matrix p1 = e.member(0).predict(inputs());
    const Matrix p2 = e.member(1).predict(inputs());
    const Matrix p = e.predict(inputs());
    for (std::size_t r = 0; r < 4; r++)
        for (std::size_t c = 0; c < 3; c++)
            cr_assert_float_eq(p(r, c), 0.25 * p1(r, c) + 0.75 * p2(r, c), 1e-15);
}

Test(ensemble, vote_shares_and_tie_rule)
{
    Ensemble e(Combination::Vote);
    e.add(member(1));
    e.add(member(2));
    const Matrix p1 = e.member(0).predict(inputs());
    const Matrix p2 = e.member(1).predict(inputs());
    const std::vector<std::size_t> c1 = classesOf(p1);
    const std::vector<std::size_t> c2 = classesOf(p2);
    const Matrix shares = e.predict(inputs());
    const std::vector<std::size_t> classes = e.classify({inputs(), inputs()});
    bool sawTie = false;
    for (std::size_t r = 0; r < 4; r++) {
        cr_assert_float_eq(shares(r, c1[r]) + (c1[r] == c2[r] ? 0 : shares(r, c2[r])), 1, 1e-15);
        if (c1[r] == c2[r]) {
            cr_assert_eq(classes[r], c1[r]);
            continue;
        }
        sawTie = true;
        // One vote each: the class with the higher mean probability wins.
        const double m1 = p1(r, c1[r]) + p2(r, c1[r]);
        const double m2 = p1(r, c2[r]) + p2(r, c2[r]);
        cr_assert_eq(classes[r], m1 >= m2 ? c1[r] : c2[r]);
    }
    cr_assert(sawTie, "the inputs should give at least one disagreement");
}

Test(ensemble, weighted_vote_follows_the_heavier_member)
{
    Ensemble e(Combination::Vote);
    e.add(member(1), 1);
    e.add(member(2), 2);
    const std::vector<std::size_t> expected = classesOf(e.member(1).predict(inputs()));
    cr_assert(e.classify({inputs(), inputs()}) == expected);
}

Test(ensemble, stack_reads_the_member_outputs)
{
    Ensemble e(Combination::Stack);
    e.add(member(1));
    e.add(member(2));
    e.setCombiner(combiner(2));
    const Matrix joined = e.memberOutputs({inputs(), inputs()});
    cr_assert_eq(joined.cols(), 6u);
    const Matrix p2 = e.member(1).predict(inputs());
    cr_assert_float_eq(joined(2, 4), p2(2, 1), 1e-15);
    Network reference = combiner(2);
    const Matrix expected = reference.predict(joined);
    const Matrix p = e.predict(inputs());
    for (std::size_t i = 0; i < 12; i++)
        cr_assert_float_eq(p.data()[i], expected.data()[i], 1e-15);
}

Test(ensemble, save_and_load_every_combination)
{
    for (const Combination c : {Combination::Average, Combination::Vote, Combination::Stack}) {
        Ensemble e(c);
        e.add(member(1), 0.5);
        e.add(member(2), 2);
        if (c == Combination::Stack)
            e.setCombiner(combiner(2));
        const Matrix before = e.predict(inputs());
        e.save(file);
        cr_assert(Ensemble::isEnsembleFile(file));
        Ensemble loaded = Ensemble::load(file);
        cr_assert_eq(loaded.combination(), c);
        cr_assert_eq(loaded.size(), 2u);
        cr_assert_eq(loaded.weight(1), 2.0);
        const Matrix after = loaded.predict(inputs());
        for (std::size_t i = 0; i < 12; i++)
            cr_assert_eq(after.data()[i], before.data()[i]);
    }
    std::remove(file.c_str());
}

Test(ensemble, a_network_file_is_not_an_ensemble)
{
    member(1).save(file);
    cr_assert_not(Ensemble::isEnsembleFile(file));
    cr_assert_throw(Ensemble::load(file), Error);
    cr_assert_not(Ensemble::isEnsembleFile("no_such_file.nn"));
    std::remove(file.c_str());
}

Test(ensemble, corrupted_and_truncated_files_are_refused)
{
    Ensemble e;
    e.add(member(1));
    e.add(member(2));
    e.save(file);
    const std::string bytes = readBytes(file);

    std::string corrupted = bytes;
    corrupted[bytes.size() / 2] ^= 0x10;
    writeBytes(file, corrupted);
    cr_assert_throw(Ensemble::load(file), Error);

    writeBytes(file, bytes.substr(0, bytes.size() - 3));
    cr_assert_throw(Ensemble::load(file), Error);

    writeBytes(file, bytes + "x");
    cr_assert_throw(Ensemble::load(file), Error);
    std::remove(file.c_str());
}

Test(ensemble, invalid_ensembles_are_refused)
{
    Ensemble e;
    cr_assert_throw(e.predict(inputs()), Error);
    cr_assert_throw(e.add(member(1), 0), Error);
    cr_assert_throw(e.add(member(1), -1), Error);
    e.add(member(1));
    cr_assert_throw(e.add(member(2, {"a", "c", "b"})), Error);
    cr_assert_throw(e.predict(std::vector<Matrix>{inputs(), inputs()}), Error);

    Ensemble stack(Combination::Stack);
    stack.add(member(1));
    stack.add(member(2));
    cr_assert_throw(stack.predict(inputs()), Error);
    // The combiner is trained on the members' outputs, so they are available before it is set.
    cr_assert_eq(stack.memberOutputs({inputs(), inputs()}).cols(), 6u);
    stack.setCombiner(combiner(3));
    cr_assert_throw(stack.predict(inputs()), Error);
    cr_assert_throw(stack.save(file), Error);

    Ensemble average;
    average.add(member(1));
    average.setCombiner(combiner(1));
    cr_assert_throw(average.predict(inputs()), Error);
}

Test(ensemble, combination_names)
{
    for (const Combination c : {Combination::Average, Combination::Vote, Combination::Stack})
        cr_assert_eq(combinationFromName(combinationName(c)), c);
    cr_assert_throw(combinationFromName("median"), Error);
}
