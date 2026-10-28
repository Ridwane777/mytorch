#include "my_torch/Search.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include "my_torch/Error.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

namespace {

// Shortest text that reads back as the same double.
std::string numberText(Scalar value)
{
    if (std::isnan(value))
        return "nan";
    char text[64];
    const auto result = std::to_chars(text, text + sizeof text, value);
    return std::string(text, result.ptr);
}

std::vector<std::string> splitCsv(const std::string &line)
{
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream in(line);
    while (std::getline(in, cell, ','))
        cells.push_back(cell);
    if (!line.empty() && line.back() == ',')
        cells.emplace_back();
    return cells;
}

std::string header(const SearchSpace &space)
{
    std::string text = "index";
    for (std::size_t i = 0; i < space.size(); i++)
        text += "," + space.name(i);
    return text + ",score";
}

std::string row(const TrialResult &result)
{
    std::string text = std::to_string(result.trial.index());
    for (const auto &[name, value] : result.trial.values())
        text += "," + toString(value);
    return text + "," + numberText(result.score);
}

bool better(Scalar a, Scalar b)
{
    return !std::isnan(a) && (std::isnan(b) || a > b);
}

// Trials already in the log, by grid index. Checks the header and every logged value.
std::map<std::size_t, Scalar> readLog(const std::string &path, const SearchSpace &space)
{
    std::map<std::size_t, Scalar> done;
    std::ifstream in(path);
    if (!in)
        return done;
    std::string line;
    if (!std::getline(in, line))
        return done;
    if (line != header(space))
        throw Error("search: '" + path + "' was written for another search space (header '" +
                    line + "')");
    for (std::size_t number = 2; std::getline(in, line); number++) {
        if (line.empty())
            continue;
        const std::vector<std::string> cells = splitCsv(line);
        const std::string where = path + ":" + std::to_string(number);
        if (cells.size() != space.size() + 2)
            throw Error("search: " + where + ": " + std::to_string(cells.size()) + " cells");
        std::size_t index = 0;
        const auto parsed = std::from_chars(cells[0].data(), cells[0].data() + cells[0].size(), index);
        if (parsed.ec != std::errc{} || parsed.ptr != cells[0].data() + cells[0].size())
            throw Error("search: " + where + ": bad index '" + cells[0] + "'");
        const Trial trial = combination(space, index);
        for (std::size_t i = 0; i < space.size(); i++)
            if (toString(trial.values()[i].second) != cells[i + 1])
                throw Error("search: " + where + ": value '" + cells[i + 1] + "' does not match");
        try {
            done[index] = std::stod(cells.back());
        } catch (const std::exception &) {
            throw Error("search: " + where + ": bad score '" + cells.back() + "'");
        }
    }
    return done;
}

} // namespace

std::string toString(const SearchValue &value)
{
    if (const Scalar *number = std::get_if<Scalar>(&value))
        return numberText(*number);
    return std::get<std::string>(value);
}

void SearchSpace::add(const std::string &name, std::vector<SearchValue> values)
{
    if (name.empty() || name.find_first_of(",\n") != std::string::npos)
        throw Error("search space: invalid hyperparameter name '" + name + "'");
    if (values.empty())
        throw Error("search space: no value for '" + name + "'");
    for (const auto &p : _params)
        if (p.first == name)
            throw Error("search space: '" + name + "' added twice");
    for (std::size_t i = 0; i < values.size(); i++) {
        const std::string text = toString(values[i]);
        if (text.empty() || text.find_first_of(",\n") != std::string::npos)
            throw Error("search space: invalid value '" + text + "' for '" + name + "'");
        for (std::size_t j = 0; j < i; j++)
            if (values[j] == values[i])
                throw Error("search space: value '" + text + "' twice for '" + name + "'");
    }
    _params.emplace_back(name, std::move(values));
}

std::size_t SearchSpace::size() const noexcept
{
    return _params.size();
}

const std::string &SearchSpace::name(std::size_t i) const
{
    return _params.at(i).first;
}

const std::vector<SearchValue> &SearchSpace::values(std::size_t i) const
{
    return _params.at(i).second;
}

std::size_t SearchSpace::combinations() const
{
    if (_params.empty())
        throw Error("search space: no hyperparameter");
    std::size_t total = 1;
    for (const auto &p : _params)
        total *= p.second.size();
    return total;
}

Trial::Trial(std::size_t index, std::vector<std::pair<std::string, SearchValue>> values)
    : _index(index), _values(std::move(values))
{
}

std::size_t Trial::index() const noexcept
{
    return _index;
}

const std::vector<std::pair<std::string, SearchValue>> &Trial::values() const noexcept
{
    return _values;
}

const SearchValue &Trial::find(const std::string &name) const
{
    for (const auto &[n, v] : _values)
        if (n == name)
            return v;
    throw Error("trial: no hyperparameter '" + name + "'");
}

Scalar Trial::number(const std::string &name) const
{
    const SearchValue &v = find(name);
    if (!std::holds_alternative<Scalar>(v))
        throw Error("trial: '" + name + "' is not a number");
    return std::get<Scalar>(v);
}

std::size_t Trial::integer(const std::string &name) const
{
    const Scalar v = number(name);
    if (!(v >= 0) || v != std::floor(v))
        throw Error("trial: '" + name + "' = " + numberText(v) + " is not a whole number");
    return static_cast<std::size_t>(v);
}

const std::string &Trial::text(const std::string &name) const
{
    const SearchValue &v = find(name);
    if (!std::holds_alternative<std::string>(v))
        throw Error("trial: '" + name + "' is not a label");
    return std::get<std::string>(v);
}

SearchResult::SearchResult(std::vector<TrialResult> trials) : _trials(std::move(trials))
{
}

const std::vector<TrialResult> &SearchResult::trials() const noexcept
{
    return _trials;
}

const TrialResult &SearchResult::best() const
{
    if (_trials.empty())
        throw Error("search: no trial");
    std::size_t best = 0;
    for (std::size_t i = 1; i < _trials.size(); i++)
        if (better(_trials[i].score, _trials[best].score))
            best = i;
    return _trials[best];
}

void SearchResult::saveCsv(const std::string &path) const
{
    std::ofstream out(path);
    if (!out)
        throw Error("cannot write '" + path + "'");
    if (_trials.empty())
        return;
    std::string text = "index";
    for (const auto &[name, value] : _trials.front().trial.values())
        text += "," + name;
    out << text << ",score\n";
    for (const TrialResult &r : _trials)
        out << row(r) << '\n';
}

Trial combination(const SearchSpace &space, std::size_t index)
{
    if (index >= space.combinations())
        throw Error("search: combination " + std::to_string(index) + " out of " +
                    std::to_string(space.combinations()));
    std::vector<std::pair<std::string, SearchValue>> values(space.size());
    std::size_t rest = index;
    for (std::size_t i = space.size(); i-- > 0;) {
        const std::size_t n = space.values(i).size();
        values[i] = {space.name(i), space.values(i)[rest % n]};
        rest /= n;
    }
    return Trial(index, std::move(values));
}

SearchResult SearchStrategy::run(const SearchSpace &space, const Objective &objective,
                                 const std::string &logPath) const
{
    const std::vector<std::size_t> chosen = choose(space);
    const std::map<std::size_t, Scalar> done =
        logPath.empty() ? std::map<std::size_t, Scalar>{} : readLog(logPath, space);
    std::vector<TrialResult> results;
    std::ofstream log;

    if (!logPath.empty()) {
        const bool fresh = !std::ifstream(logPath).good() || std::ifstream(logPath).peek() ==
                                                                 std::ifstream::traits_type::eof();
        log.open(logPath, std::ios::app);
        if (!log)
            throw Error("cannot write '" + logPath + "'");
        if (fresh)
            log << header(space) << '\n' << std::flush;
    }
    for (const std::size_t index : chosen) {
        Trial trial = combination(space, index);
        if (const auto it = done.find(index); it != done.end()) {
            results.push_back({std::move(trial), it->second});
            continue;
        }
        const Scalar score = objective(trial);
        results.push_back({std::move(trial), score});
        if (log.is_open())
            log << row(results.back()) << '\n' << std::flush;
    }
    return SearchResult(std::move(results));
}

std::vector<std::size_t> GridSearch::choose(const SearchSpace &space) const
{
    std::vector<std::size_t> all(space.combinations());
    for (std::size_t i = 0; i < all.size(); i++)
        all[i] = i;
    return all;
}

RandomSearch::RandomSearch(std::size_t count, std::uint64_t seed) : _count(count), _seed(seed)
{
    if (count == 0)
        throw Error("random search: at least one trial");
}

std::vector<std::size_t> RandomSearch::choose(const SearchSpace &space) const
{
    std::vector<std::size_t> order = permutation(space.combinations(), _seed);
    if (order.size() > _count)
        order.resize(_count);
    return order;
}

} // namespace my_torch
