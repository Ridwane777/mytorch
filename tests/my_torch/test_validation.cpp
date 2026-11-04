#include <criterion/criterion.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// 50 examples in two classes (30 and 20); the only input column holds the row number.
Dataset sampleData()
{
    Dataset d{Matrix(50, 1), Matrix(50, 2)};
    for (std::size_t r = 0; r < 50; r++) {
        d.inputs(r, 0) = static_cast<Scalar>(r);
        d.targets(r, r % 5 < 3 ? 0 : 1) = 1;
    }
    return d;
}

// The row numbers held by a source.
std::vector<std::size_t> rowsOf(const DataSource &source)
{
    std::vector<std::size_t> rows(source.size());
    std::iota(rows.begin(), rows.end(), 0);
    Matrix inputs(rows.size(), source.inputWidth());
    Matrix targets(rows.size(), source.targetWidth());
    source.fill(rows, inputs, targets);
    std::vector<std::size_t> ids;
    for (std::size_t r = 0; r < rows.size(); r++)
        ids.push_back(static_cast<std::size_t>(inputs(r, 0)));
    return ids;
}

// A report that depends on the fold: (10 + fold) of the 20 examples of class 0 are predicted right,
// and 15 of the 20 examples of class 1.
ClassificationReport reportFor(std::size_t fold)
{
    ConfusionMatrix m;
    m.counts = {{10 + fold, 10 - fold}, {5, 15}};
    return report(m);
}

} // namespace

Test(validation, summarize_gives_the_mean_and_the_sample_standard_deviation)
{
    const MetricSummary s = summarize({1, 2, 3, 4});

    cr_assert_float_eq(s.mean, 2.5, 1e-12);
    cr_assert_float_eq(s.std, std::sqrt(5.0 / 3.0), 1e-12);
    cr_assert_float_eq(s.std, 1.2910, 1e-4);
    cr_assert(s.values == (std::vector<Scalar>{1, 2, 3, 4}));
}

Test(validation, summarize_of_one_value_has_no_deviation)
{
    const MetricSummary s = summarize({0.75});

    cr_assert_float_eq(s.mean, 0.75, 1e-12);
    cr_assert_eq(s.std, 0);
}

Test(validation, summarize_of_nothing_throws)
{
    cr_assert_throw(summarize({}), Error);
}

Test(validation, cross_validate_calls_the_trainer_once_per_fold_on_complementary_parts)
{
    const Dataset data = sampleData();
    const MatrixSource source(data);
    std::vector<std::size_t> calls;
    std::vector<std::size_t> held(50, 0);

    const CrossValidation result = crossValidate(
        source, 5, 11,
        [&](const DataSource &train, const DataSource &validation, std::size_t fold) {
            calls.push_back(fold);
            cr_assert_eq(train.size() + validation.size(), 50);
            cr_assert_eq(validation.size(), 10);
            const auto trained = rowsOf(train);
            for (const std::size_t id : rowsOf(validation)) {
                held[id]++;
                cr_assert(std::find(trained.begin(), trained.end(), id) == trained.end());
            }
            return reportFor(fold);
        });

    cr_assert(calls == (std::vector<std::size_t>{0, 1, 2, 3, 4}));
    for (const std::size_t n : held)
        cr_assert_eq(n, 1, "each row must be validated exactly once");
    cr_assert_eq(result.folds.size(), 5);
}

Test(validation, cross_validate_summarizes_the_reports_in_fold_order)
{
    const Dataset data = sampleData();
    const MatrixSource source(data);
    const CrossValidation result =
        crossValidate(source, 5, 11, [](const DataSource &, const DataSource &, std::size_t fold) {
            return reportFor(fold);
        });
    std::vector<Scalar> accuracy;
    std::vector<Scalar> f1;
    for (std::size_t fold = 0; fold < 5; fold++) {
        accuracy.push_back(reportFor(fold).accuracy);
        f1.push_back(reportFor(fold).macroF1);
        cr_assert_float_eq(result.folds[fold].accuracy, accuracy.back(), 1e-12);
    }

    cr_assert(result.accuracy.values == accuracy);
    cr_assert_float_eq(result.accuracy.mean, summarize(accuracy).mean, 1e-12);
    cr_assert_float_eq(result.accuracy.std, summarize(accuracy).std, 1e-12);
    cr_assert_float_eq(result.macroF1.mean, summarize(f1).mean, 1e-12);
    cr_assert_float_eq(result.macroF1.std, summarize(f1).std, 1e-12);
    cr_assert_eq(result.macroPrecision.values.size(), 5);
    cr_assert_eq(result.macroRecall.values.size(), 5);
    cr_assert_gt(result.accuracy.std, 0);
}

Test(validation, cross_validate_rejects_a_wrong_number_of_folds)
{
    const Dataset data = sampleData();
    const MatrixSource source(data);
    const FoldTrainer trainer = [](const DataSource &, const DataSource &, std::size_t fold) {
        return reportFor(fold);
    };

    cr_assert_throw(crossValidate(source, 1, 1, trainer), Error);
    cr_assert_throw(crossValidate(source, 0, 1, trainer), Error);
    cr_assert_throw(crossValidate(source, 51, 1, trainer), Error);
    cr_assert_throw(crossValidate(source, 5, 1, FoldTrainer()), Error);
}
