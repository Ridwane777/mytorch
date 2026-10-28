#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "my_torch/Matrix.hpp"

namespace my_torch {

// Hyperparameter search (ADR-0028), on the model of Ray Tune and Optuna: the library chooses which
// combinations to try, the caller's objective trains and scores each one. Nothing here knows about
// networks: the objective is any function from a combination to a score, higher is better.

// A hyperparameter value: a number (learning rate, batch size) or a label ("256x128").
using SearchValue = std::variant<Scalar, std::string>;

std::string toString(const SearchValue &value);

// The values to try, one list per hyperparameter, in the order they were added.
class SearchSpace {
  public:
    // Throws my_torch::Error on an empty name or list, a duplicate name, or a duplicate value.
    void add(const std::string &name, std::vector<SearchValue> values);

    std::size_t size() const noexcept; // number of hyperparameters
    const std::string &name(std::size_t i) const;
    const std::vector<SearchValue> &values(std::size_t i) const;
    // Product of the list sizes; throws if the space is empty.
    std::size_t combinations() const;

  private:
    std::vector<std::pair<std::string, std::vector<SearchValue>>> _params;
};

// One combination: a value for each hyperparameter of the space.
class Trial {
  public:
    Trial(std::size_t index, std::vector<std::pair<std::string, SearchValue>> values);

    // Position of the combination in the grid order (the last hyperparameter varies fastest).
    std::size_t index() const noexcept;
    const std::vector<std::pair<std::string, SearchValue>> &values() const noexcept;
    // Throw my_torch::Error if the name is absent or the value has another type; integer also
    // throws unless the number is a non-negative whole number.
    Scalar number(const std::string &name) const;
    std::size_t integer(const std::string &name) const;
    const std::string &text(const std::string &name) const;

  private:
    const SearchValue &find(const std::string &name) const;

    std::size_t _index;
    std::vector<std::pair<std::string, SearchValue>> _values;
};

using Objective = std::function<Scalar(const Trial &)>;

struct TrialResult {
    Trial trial;
    Scalar score;
};

class SearchResult {
  public:
    explicit SearchResult(std::vector<TrialResult> trials);

    // In the order the strategy chose them, trials read back from a log included.
    const std::vector<TrialResult> &trials() const noexcept;
    // Highest score; a NaN score counts as the worst; ties go to the first one. Throws if empty.
    const TrialResult &best() const;
    // One row per trial: index, then every hyperparameter, then the score.
    void saveCsv(const std::string &path) const;

  private:
    std::vector<TrialResult> _trials;
};

// The combination of grid index `index` (last hyperparameter fastest). Throws if out of range.
Trial combination(const SearchSpace &space, std::size_t index);

// Common interface of the search methods: grid and random now; a genetic algorithm or simulated
// annealing would be other strategies on the same interface.
class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;

    // Calls the objective once per chosen combination, one after the other. With a log path, each
    // trial is appended to that CSV as soon as it ends, and the combinations already in the file
    // are not run again: a long search can be stopped and resumed. Throws if the log was written
    // for another space.
    SearchResult run(const SearchSpace &space, const Objective &objective,
                     const std::string &logPath = "") const;

  protected:
    // Grid indices of the combinations to try, in order.
    virtual std::vector<std::size_t> choose(const SearchSpace &space) const = 0;
};

// Every combination, in grid order.
class GridSearch : public SearchStrategy {
  protected:
    std::vector<std::size_t> choose(const SearchSpace &space) const override;
};

// `count` distinct combinations drawn from the grid with permutation(combinations, seed): the same
// seed gives the same draw on every toolchain. count above the grid size tries the whole grid.
class RandomSearch : public SearchStrategy {
  public:
    RandomSearch(std::size_t count, std::uint64_t seed); // throws if count is 0

  protected:
    std::vector<std::size_t> choose(const SearchSpace &space) const override;

  private:
    std::size_t _count;
    std::uint64_t _seed;
};

} // namespace my_torch
