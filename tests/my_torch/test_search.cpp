#include <criterion/criterion.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// lr {0.1, 0.01, 0.001} x batch {32, 64} x layers {"a", "b"}: 12 combinations.
SearchSpace sampleSpace()
{
    SearchSpace space;
    space.add("lr", {0.1, 0.01, 0.001});
    space.add("batch", {32.0, 64.0});
    space.add("layers", {std::string("a"), std::string("b")});
    return space;
}

// Known maximum at lr = 0.01, batch = 64, layers = "b".
Scalar score(const Trial &t)
{
    return -std::abs(std::log10(t.number("lr")) + 2) + static_cast<Scalar>(t.integer("batch")) / 64 +
           (t.text("layers") == "b" ? 1 : 0);
}

std::string tempPath(const char *name)
{
    return "/tmp/my_torch_search_" + std::to_string(getpid()) + "_" + name + ".csv";
}

std::string readAll(const std::string &path)
{
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

Test(search, grid_visits_every_combination_once_in_grid_order)
{
    std::vector<std::size_t> seen;
    const SearchResult r = GridSearch().run(sampleSpace(), [&](const Trial &t) {
        seen.push_back(t.index());
        return score(t);
    });
    cr_assert_eq(seen.size(), 12);
    for (std::size_t i = 0; i < 12; i++)
        cr_assert_eq(seen[i], i);
    // Last hyperparameter fastest: index 1 is lr 0.1, batch 32, layers "b".
    const Trial t = combination(sampleSpace(), 1);
    cr_assert_eq(t.number("lr"), 0.1);
    cr_assert_eq(t.integer("batch"), 32);
    cr_assert_eq(t.text("layers"), "b");
    cr_assert_eq(r.best().trial.number("lr"), 0.01);
    cr_assert_eq(r.best().trial.integer("batch"), 64);
    cr_assert_eq(r.best().trial.text("layers"), "b");
}

Test(search, random_search_draws_distinct_combinations_reproducibly)
{
    const auto draw = [](std::uint64_t seed, std::size_t count) {
        std::vector<std::size_t> seen;
        RandomSearch(count, seed).run(sampleSpace(), [&](const Trial &t) {
            seen.push_back(t.index());
            return 0.0;
        });
        return seen;
    };
    const auto a = draw(7, 5);
    cr_assert_eq(a.size(), 5);
    cr_assert_eq(std::set<std::size_t>(a.begin(), a.end()).size(), 5, "no combination twice");
    cr_assert(a == draw(7, 5));
    cr_assert(a != draw(8, 5));
    cr_assert_eq(draw(7, 100).size(), 12, "a count above the grid tries the whole grid");
    cr_assert_throw(RandomSearch(0, 1), Error);
}

Test(search, nan_is_never_the_best)
{
    const SearchResult r({{combination(sampleSpace(), 0), std::nan("")},
                          {combination(sampleSpace(), 1), -5},
                          {combination(sampleSpace(), 2), std::nan("")}});
    cr_assert_eq(r.best().trial.index(), 1);
}

Test(search, an_interrupted_search_resumes_from_its_log)
{
    const std::string log = tempPath("resume");
    std::remove(log.c_str());
    std::size_t calls = 0;
    const Objective failing = [&](const Trial &t) {
        if (++calls == 4)
            throw Error("stopped");
        return score(t);
    };
    cr_assert_throw(GridSearch().run(sampleSpace(), failing, log), Error);
    calls = 0;
    const Objective counting = [&](const Trial &t) {
        calls++;
        return score(t);
    };
    const SearchResult resumed = GridSearch().run(sampleSpace(), counting, log);
    cr_assert_eq(calls, 9, "only the 9 missing trials run again");
    const SearchResult fresh = GridSearch().run(sampleSpace(), score);
    cr_assert_eq(resumed.trials().size(), 12);
    for (std::size_t i = 0; i < 12; i++) {
        cr_assert_eq(resumed.trials()[i].trial.index(), fresh.trials()[i].trial.index());
        cr_assert_eq(resumed.trials()[i].score, fresh.trials()[i].score, "scores read back exactly");
    }
    // A log written for another space is refused.
    SearchSpace other;
    other.add("lr", {0.1});
    cr_assert_throw(GridSearch().run(other, score, log), Error);
    std::remove(log.c_str());
}

Test(search, save_csv_writes_one_row_per_trial)
{
    const std::string path = tempPath("save");
    RandomSearch(2, 3).run(sampleSpace(), score).saveCsv(path);
    std::istringstream text(readAll(path));
    std::string line;
    std::getline(text, line);
    cr_assert_eq(line, "index,lr,batch,layers,score");
    std::size_t rows = 0;
    while (std::getline(text, line))
        rows++;
    cr_assert_eq(rows, 2);
    std::remove(path.c_str());
    cr_assert_eq(toString(SearchValue(0.001)), "0.001");
    cr_assert_eq(toString(SearchValue(64.0)), "64");
}

Test(search, invalid_spaces_and_trials)
{
    SearchSpace space;
    cr_assert_throw(space.combinations(), Error);
    cr_assert_throw(space.add("", {1.0}), Error);
    cr_assert_throw(space.add("lr", {}), Error);
    cr_assert_throw(space.add("lr", {1.0, 1.0}), Error);
    cr_assert_throw(space.add("layers", {std::string("a,b")}), Error);
    space.add("lr", {0.0003});
    cr_assert_throw(space.add("lr", {1.0}), Error);
    const Trial t = combination(space, 0);
    cr_assert_throw(t.integer("lr"), Error);
    cr_assert_throw(t.text("lr"), Error);
    cr_assert_throw(t.number("batch"), Error);
    cr_assert_throw(combination(space, 1), Error);
}
