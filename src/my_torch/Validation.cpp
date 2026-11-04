#include "my_torch/Validation.hpp"

#include <cmath>

#include "my_torch/Data.hpp"
#include "my_torch/Error.hpp"

namespace my_torch {

MetricSummary summarize(const std::vector<Scalar> &values)
{
    if (values.empty())
        throw Error("summarize: no value");
    MetricSummary summary;
    summary.values = values;
    Scalar sum = 0;
    for (const Scalar v : values)
        sum += v;
    summary.mean = sum / static_cast<Scalar>(values.size());
    if (values.size() > 1) {
        Scalar squares = 0;
        for (const Scalar v : values)
            squares += (v - summary.mean) * (v - summary.mean);
        summary.std = std::sqrt(squares / static_cast<Scalar>(values.size() - 1));
    }
    return summary;
}

CrossValidation crossValidate(const DataSource &source, std::size_t k, std::uint64_t seed,
                              const FoldTrainer &trainer)
{
    if (!trainer)
        throw Error("cross_validate: no trainer");
    const std::vector<Fold> folds = stratifiedKFold(targetClasses(source), k, seed);
    CrossValidation result;
    for (std::size_t i = 0; i < folds.size(); i++) {
        const SubsetSource train(source, folds[i].train);
        const SubsetSource validation(source, folds[i].validation);
        result.folds.push_back(trainer(train, validation, i));
    }
    const auto summary = [&](Scalar ClassificationReport::*metric) {
        std::vector<Scalar> values;
        for (const ClassificationReport &r : result.folds)
            values.push_back(r.*metric);
        return summarize(values);
    };
    result.accuracy = summary(&ClassificationReport::accuracy);
    result.macroPrecision = summary(&ClassificationReport::macroPrecision);
    result.macroRecall = summary(&ClassificationReport::macroRecall);
    result.macroF1 = summary(&ClassificationReport::macroF1);
    return result;
}

} // namespace my_torch
