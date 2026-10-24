#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

// Classification metrics (ADR-0011). Inputs are matrices with one row per example:
// `probabilities` is the network output, `targets` the one-hot expected classes. A single-column
// matrix is read as a binary problem: class 1 when the value is >= 0.5, probability p for class 1
// and 1 - p for class 0. The predicted class is the most probable one; ties go to the lowest index.
// Every function throws my_torch::Error on empty inputs or mismatched shapes.

// Class index of each row (argmax, or >= 0.5 for a single column).
std::vector<std::size_t> classesOf(const Matrix &scores);

// counts[expected][predicted].
struct ConfusionMatrix {
    std::vector<std::vector<std::size_t>> counts;

    std::size_t classes() const noexcept;
    std::size_t total() const noexcept;
};

ConfusionMatrix confusion(const Matrix &probabilities, const Matrix &targets);

struct ClassMetrics {
    Scalar precision = 0; // 0 when the class is never predicted (predicted == 0)
    Scalar recall = 0;    // 0 when the class never occurs (support == 0)
    Scalar f1 = 0;
    std::size_t support = 0;   // examples whose expected class is this one
    std::size_t predicted = 0; // examples predicted as this class
};

struct ClassificationReport {
    ConfusionMatrix confusion;
    std::vector<ClassMetrics> perClass;
    Scalar accuracy = 0;
    Scalar macroPrecision = 0; // plain mean over classes
    Scalar macroRecall = 0;
    Scalar macroF1 = 0;
    Scalar weightedPrecision = 0; // mean weighted by support
    Scalar weightedRecall = 0;
    Scalar weightedF1 = 0;
    Scalar logLoss = 0; // mean -log p(expected class), p bounded below by 1e-15
};

ClassificationReport report(const Matrix &probabilities, const Matrix &targets);

// Fraction of rows whose expected class is among the k most probable ones. Throws unless
// 1 <= k <= number of classes.
Scalar topKAccuracy(const Matrix &probabilities, const Matrix &targets, std::size_t k);

// One-vs-rest area under the ROC curve of each class, by average ranks (ties count 1/2). NaN for a
// class that is always or never expected, where the AUC is undefined.
std::vector<Scalar> rocAuc(const Matrix &probabilities, const Matrix &targets);

struct CalibrationBin {
    std::size_t count = 0;
    Scalar confidence = 0; // mean top probability of the rows in the bin
    Scalar accuracy = 0;   // fraction of those rows predicted right
};

struct Calibration {
    std::vector<CalibrationBin> bins; // equal-width bins of the top probability over [0, 1]
    Scalar expectedCalibrationError = 0; // Σ (count / total)·|accuracy - confidence|
};

Calibration calibration(const Matrix &probabilities, const Matrix &targets, std::size_t bins = 10);

// History as CSV, one line per epoch. Throws my_torch::Error if the file cannot be written.
void writeHistoryCsv(const History &history, const std::string &path);
// Report as JSON; classNames (optional) labels the classes, otherwise they are numbered.
void writeReportJson(const ClassificationReport &report, const std::string &path,
                     const std::vector<std::string> &classNames = {});

} // namespace my_torch
