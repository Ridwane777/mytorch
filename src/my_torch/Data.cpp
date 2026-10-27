#include "my_torch/Data.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "my_torch/Error.hpp"
#include "my_torch/Metrics.hpp"

namespace my_torch {

namespace {

// The indices of each class, each list in the order of one shared shuffle.
std::vector<std::vector<std::size_t>> shuffledByClass(const std::vector<std::size_t> &classes,
                                                      std::uint64_t seed)
{
    const std::size_t count =
        classes.empty() ? 0 : *std::max_element(classes.begin(), classes.end()) + 1;
    std::vector<std::vector<std::size_t>> byClass(count);

    for (const std::size_t i : permutation(classes.size(), seed))
        byClass[classes[i]].push_back(i);
    return byClass;
}

} // namespace

std::vector<std::vector<std::size_t>> stratifiedSplit(const std::vector<std::size_t> &classes,
                                                      const std::vector<Scalar> &proportions,
                                                      std::uint64_t seed)
{
    if (classes.empty())
        throw Error("stratified_split: no example");
    if (proportions.empty())
        throw Error("stratified_split: no proportion");
    Scalar sum = 0;
    for (const Scalar p : proportions) {
        if (!(p > 0))
            throw Error("stratified_split: proportions must be positive");
        sum += p;
    }
    if (std::abs(sum - 1) > 1e-9)
        throw Error("stratified_split: proportions sum to " + std::to_string(sum) + ", not 1");
    const std::size_t parts = proportions.size();
    std::vector<std::vector<std::size_t>> out(parts);

    for (const auto &members : shuffledByClass(classes, seed)) {
        const std::size_t n = members.size();
        std::vector<std::size_t> sizes(parts);
        std::vector<Scalar> remainders(parts);
        std::size_t given = 0;
        for (std::size_t j = 0; j < parts; j++) {
            const Scalar exact = proportions[j] * static_cast<Scalar>(n);
            sizes[j] = static_cast<std::size_t>(std::floor(exact));
            remainders[j] = exact - static_cast<Scalar>(sizes[j]);
            given += sizes[j];
        }
        std::vector<std::size_t> order(parts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return remainders[a] > remainders[b]; });
        for (std::size_t r = 0; given < n; r++, given++)
            sizes[order[r % parts]]++;
        std::size_t next = 0;
        for (std::size_t j = 0; j < parts; j++)
            for (std::size_t i = 0; i < sizes[j]; i++)
                out[j].push_back(members[next++]);
    }
    for (auto &part : out)
        std::sort(part.begin(), part.end());
    return out;
}

std::vector<Fold> stratifiedKFold(const std::vector<std::size_t> &classes, std::size_t k,
                                  std::uint64_t seed)
{
    if (k < 2 || k > classes.size())
        throw Error("stratified_k_fold: k must be between 2 and the number of examples (" +
                    std::to_string(classes.size()) + "), got " + std::to_string(k));
    std::vector<std::size_t> foldOf(classes.size());
    std::size_t turn = 0;

    // Dealt in turn, continuing from one class to the next: fold sizes differ by at most one.
    for (const auto &members : shuffledByClass(classes, seed))
        for (const std::size_t i : members)
            foldOf[i] = turn++ % k;
    std::vector<Fold> folds(k);
    for (std::size_t i = 0; i < classes.size(); i++)
        for (std::size_t f = 0; f < k; f++)
            (f == foldOf[i] ? folds[f].validation : folds[f].train).push_back(i);
    return folds;
}

SubsetSource::SubsetSource(const DataSource &source, std::vector<std::size_t> indices)
    : _source(source), _indices(std::move(indices))
{
    for (const std::size_t i : _indices)
        if (i >= _source.size())
            throw Error("subset: index " + std::to_string(i) + " out of a source of " +
                        std::to_string(_source.size()) + " rows");
}

std::size_t SubsetSource::size() const
{
    return _indices.size();
}

std::size_t SubsetSource::inputWidth() const
{
    return _source.inputWidth();
}

std::size_t SubsetSource::targetWidth() const
{
    return _source.targetWidth();
}

void SubsetSource::fill(const std::vector<std::size_t> &rows, Matrix &inputs,
                        Matrix &targets) const
{
    std::vector<std::size_t> mapped(rows.size());
    for (std::size_t r = 0; r < rows.size(); r++) {
        if (rows[r] >= _indices.size())
            throw Error("subset: row " + std::to_string(rows[r]) + " out of " +
                        std::to_string(_indices.size()));
        mapped[r] = _indices[rows[r]];
    }
    _source.fill(mapped, inputs, targets);
}

std::vector<std::size_t> targetClasses(const DataSource &source)
{
    std::vector<std::size_t> classes;
    classes.reserve(source.size());
    for (std::size_t begin = 0; begin < source.size(); begin += EVALUATION_CHUNK) {
        const std::size_t end = std::min(begin + EVALUATION_CHUNK, source.size());
        std::vector<std::size_t> rows(end - begin);
        std::iota(rows.begin(), rows.end(), begin);
        Matrix inputs(rows.size(), source.inputWidth());
        Matrix targets(rows.size(), source.targetWidth());
        source.fill(rows, inputs, targets);
        for (const std::size_t c : classesOf(targets))
            classes.push_back(c);
    }
    return classes;
}

} // namespace my_torch
