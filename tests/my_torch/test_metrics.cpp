#include <criterion/criterion.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

struct Cell {
    std::size_t expected;
    std::size_t predicted;
    std::size_t count;
};

// Rows with probability 0.7 on the predicted class and 0.3 shared by the others; one-hot targets.
struct Data {
    Matrix probabilities;
    Matrix targets;
};

Data fromCells(const std::vector<Cell> &cells, std::size_t classes)
{
    std::size_t rows = 0;
    for (const Cell &c : cells)
        rows += c.count;
    Data d{Matrix(rows, classes), Matrix(rows, classes)};
    std::size_t r = 0;
    for (const Cell &c : cells)
        for (std::size_t i = 0; i < c.count; i++, r++) {
            for (std::size_t k = 0; k < classes; k++)
                d.probabilities(r, k) = 0.3 / static_cast<Scalar>(classes - 1);
            d.probabilities(r, c.predicted) = 0.7;
            d.targets(r, c.expected) = 1;
        }
    return d;
}

// The 3-class example of the explanation: rows are expected Nothing / Check / Mate.
//   [8 2 0]
//   [1 5 1]
//   [0 1 2]
Data example()
{
    return fromCells({{0, 0, 8}, {0, 1, 2}, {1, 0, 1}, {1, 1, 5}, {1, 2, 1}, {2, 1, 1}, {2, 2, 2}},
                     3);
}

std::string readFile(const std::string &path)
{
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

} // namespace

Test(metrics, confusion_matrix_of_the_example)
{
    const Data d = example();
    const ConfusionMatrix m = confusion(d.probabilities, d.targets);

    cr_assert_eq(m.classes(), 3);
    cr_assert_eq(m.total(), 20);
    cr_assert_eq(m.counts[0][0], 8);
    cr_assert_eq(m.counts[0][1], 2);
    cr_assert_eq(m.counts[1][2], 1);
    cr_assert_eq(m.counts[2][1], 1);
    cr_assert_eq(m.counts[2][2], 2);
}

Test(metrics, report_of_the_example_computed_by_hand)
{
    const Data d = example();
    const ClassificationReport r = report(d.probabilities, d.targets);
    const Scalar eps = 1e-12;

    cr_assert_float_eq(r.accuracy, 15.0 / 20, eps);
    // Nothing: precision 8/9 (column), recall 8/10 (row).
    cr_assert_float_eq(r.perClass[0].precision, 8.0 / 9, eps);
    cr_assert_float_eq(r.perClass[0].recall, 8.0 / 10, eps);
    cr_assert_float_eq(r.perClass[0].f1, 2 * (8.0 / 9) * 0.8 / (8.0 / 9 + 0.8), eps);
    // Check: precision 5/8, recall 5/7, F1 2/3.
    cr_assert_float_eq(r.perClass[1].precision, 5.0 / 8, eps);
    cr_assert_float_eq(r.perClass[1].recall, 5.0 / 7, eps);
    cr_assert_float_eq(r.perClass[1].f1, 2.0 / 3, eps);
    // Mate: precision 2/3, recall 2/3.
    cr_assert_float_eq(r.perClass[2].precision, 2.0 / 3, eps);
    cr_assert_float_eq(r.perClass[2].recall, 2.0 / 3, eps);
    cr_assert_eq(r.perClass[0].support, 10);
    cr_assert_eq(r.perClass[1].predicted, 8);

    const Scalar f1Nothing = 2 * (8.0 / 9) * 0.8 / (8.0 / 9 + 0.8);
    cr_assert_float_eq(r.macroPrecision, (8.0 / 9 + 5.0 / 8 + 2.0 / 3) / 3, eps);
    cr_assert_float_eq(r.macroRecall, (0.8 + 5.0 / 7 + 2.0 / 3) / 3, eps);
    cr_assert_float_eq(r.macroF1, (f1Nothing + 2.0 / 3 + 2.0 / 3) / 3, eps);
    cr_assert_float_eq(r.weightedPrecision, (10 * 8.0 / 9 + 7 * 5.0 / 8 + 3 * 2.0 / 3) / 20, eps);
    // Weighted recall is always the accuracy.
    cr_assert_float_eq(r.weightedRecall, r.accuracy, eps);
    cr_assert_float_eq(r.weightedF1, (10 * f1Nothing + 7 * 2.0 / 3 + 3 * 2.0 / 3) / 20, eps);
    // 15 rows give 0.7 to the right class, 5 rows give it 0.15.
    cr_assert_float_eq(r.logLoss, (15 * -std::log(0.7) + 5 * -std::log(0.15)) / 20, eps);
}

Test(metrics, lazy_network_has_high_accuracy_but_zero_recall_on_the_rare_class)
{
    // Always answers class 0: 90 right, the 10 examples of class 1 all missed.
    const Data d = fromCells({{0, 0, 90}, {1, 0, 10}}, 2);
    const ClassificationReport r = report(d.probabilities, d.targets);

    cr_assert_float_eq(r.accuracy, 0.9, 1e-12);
    cr_assert_eq(r.perClass[1].recall, 0);
    cr_assert_eq(r.perClass[1].f1, 0);
    // Class 1 is never predicted: precision 0 by convention, not NaN.
    cr_assert_eq(r.perClass[1].predicted, 0);
    cr_assert_eq(r.perClass[1].precision, 0);
    cr_assert_lt(r.macroF1, 0.5);
}

Test(metrics, ties_go_to_the_lowest_class_and_one_column_is_binary)
{
    Matrix tie(1, 3, 1.0 / 3);
    cr_assert_eq(classesOf(tie)[0], 0);

    Matrix single(3, 1);
    single(0, 0) = 0.2;
    single(1, 0) = 0.5;
    single(2, 0) = 0.9;
    const auto c = classesOf(single);
    cr_assert_eq(c[0], 0);
    cr_assert_eq(c[1], 1);
    cr_assert_eq(c[2], 1);
    cr_assert_eq(confusion(single, single).classes(), 2);
}

Test(metrics, top_k_accuracy)
{
    const Data d = example();
    cr_assert_float_eq(topKAccuracy(d.probabilities, d.targets, 1), 0.75, 1e-12);
    cr_assert_float_eq(topKAccuracy(d.probabilities, d.targets, 3), 1, 1e-12);

    // Expected class 2 ranked second in row 0, third in row 1.
    Matrix p(2, 3);
    p(0, 0) = 0.5;
    p(0, 1) = 0.1;
    p(0, 2) = 0.4;
    p(1, 0) = 0.5;
    p(1, 1) = 0.3;
    p(1, 2) = 0.2;
    Matrix t(2, 3);
    t(0, 2) = 1;
    t(1, 2) = 1;
    cr_assert_float_eq(topKAccuracy(p, t, 2), 0.5, 1e-12);
    cr_assert_throw(topKAccuracy(p, t, 0), Error);
    cr_assert_throw(topKAccuracy(p, t, 4), Error);
}

Test(metrics, roc_auc_by_hand)
{
    // Class 1 scores: positives 0.9 and 0.6, negatives 0.6 and 0.2. Of the 4 pairs, 3 are ranked
    // right and 1 is a tie (1/2): AUC = 3.5 / 4.
    Matrix p(4, 2);
    Matrix t(4, 2);
    const Scalar scores[] = {0.9, 0.6, 0.6, 0.2};
    const int positive[] = {1, 1, 0, 0};
    for (std::size_t r = 0; r < 4; r++) {
        p(r, 1) = scores[r];
        p(r, 0) = 1 - scores[r];
        t(r, positive[r] ? 1 : 0) = 1;
    }
    const auto auc = rocAuc(p, t);
    cr_assert_float_eq(auc[1], 0.875, 1e-12);
    cr_assert_float_eq(auc[0], 0.875, 1e-12);
}

Test(metrics, roc_auc_extremes_and_undefined_class)
{
    Matrix p(4, 2);
    Matrix t(4, 2);
    for (std::size_t r = 0; r < 4; r++) {
        p(r, 1) = r < 2 ? 0.8 : 0.3;
        p(r, 0) = 1 - p(r, 1);
        t(r, r < 2 ? 1 : 0) = 1;
    }
    cr_assert_float_eq(rocAuc(p, t)[1], 1, 1e-12);
    // Same score for everyone: every pair is a tie, AUC 0.5.
    Matrix flat(4, 2, 0.5);
    cr_assert_float_eq(rocAuc(flat, t)[1], 0.5, 1e-12);
    // A class that never occurs has no AUC.
    Matrix three(4, 3, 1.0 / 3);
    Matrix t3(4, 3);
    for (std::size_t r = 0; r < 4; r++)
        t3(r, r % 2) = 1;
    cr_assert(std::isnan(rocAuc(three, t3)[2]));
}

Test(metrics, roc_auc_matches_the_pairwise_definition)
{
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> coarse(0, 9); // many ties on purpose
    const std::size_t n = 200;
    Matrix p(n, 2);
    Matrix t(n, 2);
    for (std::size_t r = 0; r < n; r++) {
        p(r, 1) = coarse(rng) / 10.0;
        p(r, 0) = 1 - p(r, 1);
        t(r, coarse(rng) < 4 ? 1 : 0) = 1;
    }
    Scalar wins = 0;
    Scalar pairs = 0;
    for (std::size_t a = 0; a < n; a++)
        for (std::size_t b = 0; b < n; b++)
            if (t(a, 1) == 1 && t(b, 1) == 0) {
                pairs++;
                wins += p(a, 1) > p(b, 1) ? 1 : (p(a, 1) == p(b, 1) ? 0.5 : 0);
            }
    cr_assert_float_eq(rocAuc(p, t)[1], wins / pairs, 1e-12);
}

Test(metrics, calibration_by_hand)
{
    // Ten rows at confidence 0.8 with 8 right: perfectly calibrated.
    Matrix p(10, 2);
    Matrix t(10, 2);
    for (std::size_t r = 0; r < 10; r++) {
        p(r, 0) = 0.8;
        p(r, 1) = 0.2;
        t(r, r < 8 ? 0 : 1) = 1;
    }
    Calibration c = calibration(p, t, 10);
    cr_assert_eq(c.bins[8].count, 10);
    cr_assert_float_eq(c.bins[8].confidence, 0.8, 1e-12);
    cr_assert_float_eq(c.bins[8].accuracy, 0.8, 1e-12);
    cr_assert_float_eq(c.expectedCalibrationError, 0, 1e-12);

    // Overconfident: says 0.9, right 6 times out of 10, so ECE = |0.6 - 0.9|.
    for (std::size_t r = 0; r < 10; r++) {
        p(r, 0) = 0.9;
        p(r, 1) = 0.1;
        t(r, 0) = r < 6 ? 1 : 0;
        t(r, 1) = r < 6 ? 0 : 1;
    }
    c = calibration(p, t, 10);
    cr_assert_eq(c.bins[9].count, 10);
    cr_assert_float_eq(c.expectedCalibrationError, 0.3, 1e-12);
    cr_assert_throw(calibration(p, t, 0), Error);
}

Test(metrics, rejects_mismatched_or_empty_inputs)
{
    cr_assert_throw(confusion(Matrix(2, 3), Matrix(2, 2)), Error);
    cr_assert_throw(report(Matrix(0, 3), Matrix(0, 3)), Error);
    cr_assert_throw(rocAuc(Matrix(2, 2), Matrix(3, 2)), Error);
}

Test(metrics, exports_history_csv)
{
    History h(2);
    h[0].trainLoss = 0.5;
    h[0].validationAccuracy = 0.25;
    h[1].trainAccuracy = 1;
    const std::string path = "metrics_history_test.csv";

    writeHistoryCsv(h, path);
    const std::string text = readFile(path);
    std::remove(path.c_str());
    cr_assert_str_eq(text.c_str(),
                     "epoch,train_loss,validation_loss,penalty,train_accuracy,validation_accuracy\n"
                     "0,0.5,0,0,0,0.25\n"
                     "1,0,0,0,1,0\n");
    cr_assert_throw(writeHistoryCsv(h, "/nonexistent/dir/h.csv"), Error);
}

Test(metrics, exports_report_json)
{
    const Data d = example();
    const ClassificationReport r = report(d.probabilities, d.targets);
    const std::string path = "metrics_report_test.json";

    writeReportJson(r, path, {"Nothing", "Check", "Mate"});
    const std::string text = readFile(path);
    std::remove(path.c_str());
    cr_assert(text.find("\"accuracy\": 0.75") != std::string::npos, "%s", text.c_str());
    cr_assert(text.find("{\"name\": \"Check\", \"precision\": 0.625") != std::string::npos);
    cr_assert(text.find("\"support\": 10, \"predicted\": 9") != std::string::npos);
    cr_assert(text.find("[8, 2, 0]") != std::string::npos);
    cr_assert(text.find("[0, 1, 2]") != std::string::npos);
}

Test(metrics, fit_records_train_and_validation_accuracy)
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
    CrossEntropy loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 500;
    config.batchSize = 4;
    Dataset data{x, t};

    History h = net.fit(data, &data, loss, adam, config);
    cr_assert_float_eq(h.back().trainAccuracy, 1, 1e-12);
    cr_assert_float_eq(h.back().validationAccuracy, 1, 1e-12);
    cr_assert_float_eq(h.back().validationAccuracy, report(net.predict(x), t).accuracy, 1e-12);
}

// Class indices of the same cells, as a predictions file would give them back.
static void indicesOf(const std::vector<Cell> &cells, std::vector<std::size_t> &expected,
                      std::vector<std::size_t> &predicted)
{
    for (const Cell &c : cells)
        for (std::size_t i = 0; i < c.count; i++) {
            expected.push_back(c.expected);
            predicted.push_back(c.predicted);
        }
}

Test(metrics, confusion_from_indices_equals_confusion_from_probabilities)
{
    const std::vector<Cell> cells = {{0, 0, 5}, {0, 2, 1}, {1, 1, 3}, {2, 0, 2}, {2, 2, 4}};
    const Data d = fromCells(cells, 3);
    std::vector<std::size_t> expected, predicted;
    indicesOf(cells, expected, predicted);

    cr_assert(confusion(expected, predicted, 3).counts == confusion(d.probabilities, d.targets).counts);
}

Test(metrics, report_from_confusion_equals_report_from_probabilities)
{
    // Class 3 is never predicted and class 4 never occurs: both give 0, not a division by zero.
    const std::vector<Cell> cells = {{0, 0, 7}, {0, 1, 2}, {1, 1, 4}, {1, 2, 1}, {2, 2, 3},
                                     {3, 0, 2}, {3, 2, 1}, {0, 4, 1}};
    const Data d = fromCells(cells, 5);
    std::vector<std::size_t> expected, predicted;
    indicesOf(cells, expected, predicted);
    const ClassificationReport a = report(d.probabilities, d.targets);
    const ClassificationReport b = report(confusion(expected, predicted, 5));

    // Same code path, so the numbers are equal bit for bit, not only close.
    cr_assert_eq(a.accuracy, b.accuracy);
    cr_assert_eq(a.macroF1, b.macroF1);
    cr_assert_eq(a.weightedF1, b.weightedF1);
    for (std::size_t c = 0; c < 5; c++) {
        cr_assert_eq(a.perClass[c].precision, b.perClass[c].precision);
        cr_assert_eq(a.perClass[c].recall, b.perClass[c].recall);
        cr_assert_eq(a.perClass[c].f1, b.perClass[c].f1);
        cr_assert_eq(a.perClass[c].support, b.perClass[c].support);
    }
    cr_assert_eq(b.perClass[3].precision, 0);
    cr_assert_eq(b.perClass[4].recall, 0);
    cr_assert_eq(b.logLoss, 0);
}

Test(metrics, merge_into_families)
{
    // Classes: Nothing, Check White, Check Black, Checkmate White, Checkmate Black.
    ConfusionMatrix m{{{10, 1, 0, 0, 0},
                       {0, 8, 0, 2, 0},
                       {1, 0, 7, 0, 3},
                       {0, 1, 0, 6, 0},
                       {0, 0, 2, 0, 5}}};
    const ConfusionMatrix families = merge(m, {0, 1, 1, 2, 2}, 3);

    cr_assert(families.counts == (std::vector<std::vector<std::size_t>>{{10, 1, 0}, {1, 15, 5},
                                                                        {0, 3, 11}}));
    cr_assert_eq(families.total(), m.total());
}

Test(metrics, label_based_errors)
{
    cr_assert_throw(confusion(std::vector<std::size_t>{0, 1}, std::vector<std::size_t>{0}, 2),
                    Error);
    cr_assert_throw(confusion(std::vector<std::size_t>{}, std::vector<std::size_t>{}, 2), Error);
    cr_assert_throw(confusion(std::vector<std::size_t>{0, 2}, std::vector<std::size_t>{0, 1}, 2),
                    Error);
    const ConfusionMatrix m{{{1, 0}, {0, 1}}};
    cr_assert_throw(merge(m, {0}, 1), Error);
    cr_assert_throw(merge(m, {0, 2}, 2), Error);
    cr_assert_throw(report(ConfusionMatrix{}), Error);
    cr_assert_throw(report(ConfusionMatrix{{{0, 0}, {0, 0}}}), Error);
}
