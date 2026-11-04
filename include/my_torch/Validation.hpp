#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "my_torch/Matrix.hpp"
#include "my_torch/Metrics.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

// Stratified k-fold cross-validation (ADR-0039), on the model of scikit-learn's
// cross_validate: the folds come from stratifiedKFold (ADR-0027), the training and the scoring of
// each fold are left to a callback, and every metric is reported as mean ± standard deviation over
// the folds. Every function throws my_torch::Error on invalid arguments.

// The values of one metric over the folds, and their summary.
struct MetricSummary {
    Scalar mean = 0;
    Scalar std = 0; // sample standard deviation (n - 1 in the denominator), 0 for one value
    std::vector<Scalar> values;
};

// Throws if values is empty.
MetricSummary summarize(const std::vector<Scalar> &values);

struct CrossValidation {
    std::vector<ClassificationReport> folds; // in fold order
    MetricSummary accuracy;
    MetricSummary macroPrecision;
    MetricSummary macroRecall;
    MetricSummary macroF1;
};

// Trains a fresh model on `train` and scores it on `validation`, the held-out part of the fold
// (fold counts from 0). Both sources are views on the source given to crossValidate and are only
// valid during the call.
using FoldTrainer = std::function<ClassificationReport(
    const DataSource &train, const DataSource &validation, std::size_t fold)>;

// Splits the source into k stratified folds (seed as in stratifiedKFold) and calls the trainer
// once per fold, in order, one fold after the other. Needs 2 <= k <= source.size(). The source
// must outlive the call.
CrossValidation crossValidate(const DataSource &source, std::size_t k, std::uint64_t seed,
                              const FoldTrainer &trainer);

} // namespace my_torch
