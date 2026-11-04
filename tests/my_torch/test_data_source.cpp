#include <criterion/criterion.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// A source that builds each row from a formula, never holding the examples as a matrix: the
// kind of source the analyzer uses. Row i of example k is derived from k only.
class FormulaSource : public DataSource {
  public:
    FormulaSource(std::size_t count, std::size_t first) : _count(count), _first(first)
    {
    }

    std::size_t size() const override
    {
        return _count;
    }
    std::size_t inputWidth() const override
    {
        return 3;
    }
    std::size_t targetWidth() const override
    {
        return 2;
    }
    void fill(const std::vector<std::size_t> &rows, Matrix &inputs, Matrix &targets) const override
    {
        for (std::size_t r = 0; r < rows.size(); r++) {
            const std::size_t k = _first + rows[r];
            for (std::size_t c = 0; c < 3; c++)
                inputs(r, c) = std::sin(static_cast<Scalar>(k * 3 + c));
            targets(r, (k * 7) % 3 == 0 ? 1 : 0) = 1;
        }
    }

    // The same examples as matrices, for the reference run.
    Dataset materialize() const
    {
        Dataset d{Matrix(_count, 3), Matrix(_count, 2)};
        std::vector<std::size_t> all(_count);
        for (std::size_t i = 0; i < _count; i++)
            all[i] = i;
        fill(all, d.inputs, d.targets);
        return d;
    }

  private:
    std::size_t _count;
    std::size_t _first;
};

Network smallNet()
{
    Network net;
    net.add(std::make_unique<Dense>(3, 6, 1));
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(6, 2, 2));
    net.add(std::make_unique<Softmax>());
    return net;
}

std::vector<Scalar> weightsOf(Network &net)
{
    std::vector<Scalar> values;
    for (const Parameter &p : net.parameters())
        values.insert(values.end(), p.value->data(), p.value->data() + p.value->size());
    return values;
}

} // namespace

Test(data_source, a_source_trains_exactly_like_matrices)
{
    // 50 training examples in batches of 8 (the last batch is partial), shuffled, with a
    // validation set and early stopping: every path of fit is used.
    const FormulaSource train(50, 0);
    const FormulaSource validation(20, 1000);
    const Dataset trainMatrices = train.materialize();
    const Dataset validationMatrices = validation.materialize();
    CrossEntropy loss;
    TrainConfig config;
    config.epochs = 30;
    config.batchSize = 8;
    config.patience = 5;

    Network fromSource = smallNet();
    Adam a(0.05);
    const History h1 = fromSource.fit(train, &validation, loss, a, config);
    Network fromMatrices = smallNet();
    Adam b(0.05);
    const History h2 = fromMatrices.fit(trainMatrices, &validationMatrices, loss, b, config);

    cr_assert_eq(h1.size(), h2.size());
    for (std::size_t e = 0; e < h1.size(); e++) {
        cr_assert_eq(h1[e].trainLoss, h2[e].trainLoss, "epoch %zu", e);
        cr_assert_eq(h1[e].validationLoss, h2[e].validationLoss, "epoch %zu", e);
        cr_assert_eq(h1[e].trainAccuracy, h2[e].trainAccuracy);
    }
    cr_assert(weightsOf(fromSource) == weightsOf(fromMatrices));
}

Test(data_source, validation_larger_than_one_chunk)
{
    const std::size_t count = EVALUATION_CHUNK + 904;
    const FormulaSource validation(count, 0);
    const Dataset all = validation.materialize();
    Network net = smallNet();
    CrossEntropy loss;
    SGD frozen(1);
    frozen.setLearningRate(0);
    TrainConfig config;
    config.epochs = 1;

    const History h = net.fit(FormulaSource(10, 0), &validation, loss, frozen, config);
    const Matrix predicted = net.predict(all.inputs);
    cr_assert_float_eq(h[0].validationLoss, loss.value(predicted, all.targets), 1e-12);
    cr_assert_eq(h[0].validationAccuracy, report(predicted, all.targets).accuracy);
}

Test(data_source, rejects_bad_sources)
{
    Network net = smallNet();
    CrossEntropy loss;
    Adam adam;
    TrainConfig config;

    cr_assert_throw(net.fit(FormulaSource(0, 0), nullptr, loss, adam, config), Error);
    Dataset wide{Matrix(5, 4), Matrix(5, 2)};
    const MatrixSource wideSource(wide);
    const FormulaSource train(5, 0);
    cr_assert_throw(net.fit(train, &wideSource, loss, adam, config), Error);
    Dataset uneven{Matrix(5, 3), Matrix(4, 2)};
    cr_assert_throw(MatrixSource{uneven}, Error);
}
