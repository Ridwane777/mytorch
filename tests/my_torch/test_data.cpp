#include <criterion/criterion.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "gradient_check.hpp"
#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

// 103 examples in three unbalanced classes (58, 30 and 15 examples).
std::vector<std::size_t> sampleClasses()
{
    std::vector<std::size_t> classes;
    for (std::size_t i = 0; i < 103; i++)
        classes.push_back(i % 7 == 0 ? 2 : (i % 3 == 0 ? 1 : 0));
    return classes;
}

std::size_t countOf(const std::vector<std::size_t> &indices, const std::vector<std::size_t> &classes,
                    std::size_t c)
{
    return static_cast<std::size_t>(
        std::count_if(indices.begin(), indices.end(), [&](std::size_t i) { return classes[i] == c; }));
}

std::size_t totalOf(const std::vector<std::size_t> &classes, std::size_t c)
{
    return static_cast<std::size_t>(std::count(classes.begin(), classes.end(), c));
}

} // namespace

Test(data, stratified_split_is_a_partition_with_class_proportions)
{
    const auto classes = sampleClasses();
    const std::vector<Scalar> proportions = {0.8, 0.1, 0.1};
    const auto parts = stratifiedSplit(classes, proportions, 42);

    std::vector<std::size_t> all;
    for (const auto &part : parts) {
        cr_assert(std::is_sorted(part.begin(), part.end()));
        all.insert(all.end(), part.begin(), part.end());
    }
    std::sort(all.begin(), all.end());
    std::vector<std::size_t> expected(classes.size());
    for (std::size_t i = 0; i < expected.size(); i++)
        expected[i] = i;
    cr_assert(all == expected, "every index must be in exactly one part");
    for (std::size_t c = 0; c < 3; c++)
        for (std::size_t j = 0; j < 3; j++) {
            const double exact = proportions[j] * static_cast<double>(totalOf(classes, c));
            const double got = static_cast<double>(countOf(parts[j], classes, c));
            cr_assert(got >= exact - 1 && got <= exact + 1, "class %zu, part %zu: %f for %f", c, j,
                      got, exact);
        }
}

Test(data, stratified_split_depends_only_on_the_seed)
{
    const auto classes = sampleClasses();
    cr_assert(stratifiedSplit(classes, {0.5, 0.5}, 7) == stratifiedSplit(classes, {0.5, 0.5}, 7));
    cr_assert(stratifiedSplit(classes, {0.5, 0.5}, 7) != stratifiedSplit(classes, {0.5, 0.5}, 8));
}

Test(data, k_fold_validation_parts_cover_every_index_once)
{
    const auto classes = sampleClasses();
    const auto folds = stratifiedKFold(classes, 5, 3);
    std::vector<std::size_t> seen(classes.size(), 0);

    cr_assert_eq(folds.size(), 5);
    for (const Fold &f : folds) {
        cr_assert_eq(f.train.size() + f.validation.size(), classes.size());
        for (const std::size_t i : f.validation) {
            seen[i]++;
            cr_assert(!std::binary_search(f.train.begin(), f.train.end(), i));
        }
        // Each fold holds a fifth of each class, within one example.
        for (std::size_t c = 0; c < 3; c++) {
            const double exact = static_cast<double>(totalOf(classes, c)) / 5;
            const double got = static_cast<double>(countOf(f.validation, classes, c));
            cr_assert(got >= exact - 1 && got <= exact + 1);
        }
        cr_assert(f.validation.size() == 20 || f.validation.size() == 21);
    }
    for (const std::size_t n : seen)
        cr_assert_eq(n, 1);
    cr_assert(stratifiedKFold(classes, 5, 3)[2].validation == folds[2].validation);
}

Test(data, subset_source_trains_like_a_copy_of_its_rows)
{
    Matrix x = gradient_check::random(120, 6, 5);
    Matrix t(120, 3);
    for (std::size_t r = 0; r < 120; r++)
        t(r, (r * 7) % 3) = 1;
    const Dataset all{x, t};
    const MatrixSource source(all);
    const auto folds = stratifiedKFold(targetClasses(source), 4, 9);
    const std::vector<std::size_t> &rows = folds[1].train;
    Dataset copy{Matrix(rows.size(), 6), Matrix(rows.size(), 3)};
    for (std::size_t r = 0; r < rows.size(); r++) {
        for (std::size_t c = 0; c < 6; c++)
            copy.inputs(r, c) = x(rows[r], c);
        for (std::size_t c = 0; c < 3; c++)
            copy.targets(r, c) = t(rows[r], c);
    }
    const auto train = [](const DataSource &data) {
        Network net;
        net.add(std::make_unique<Dense>(6, 8, 1));
        net.add(std::make_unique<ReLU>());
        net.add(std::make_unique<Dense>(8, 3, 2));
        net.add(std::make_unique<Softmax>());
        CrossEntropy loss;
        Adam adam(0.01);
        TrainConfig config;
        config.epochs = 3;
        config.batchSize = 16;
        net.fit(data, nullptr, loss, adam, config);
        std::vector<Scalar> w;
        for (const Parameter &p : net.parameters())
            w.insert(w.end(), p.value->data(), p.value->data() + p.value->size());
        return w;
    };
    const MatrixSource copied(copy);
    cr_assert(train(SubsetSource(source, rows)) == train(copied),
              "a subset must train exactly like a copy of its rows");
}

Test(data, target_classes_reads_the_argmax_of_each_target)
{
    Dataset d{Matrix(5000, 1), Matrix(5000, 4)};
    for (std::size_t r = 0; r < 5000; r++)
        d.targets(r, r % 4) = 1;
    const auto classes = targetClasses(MatrixSource(d));
    cr_assert_eq(classes.size(), 5000);
    for (std::size_t r = 0; r < 5000; r++)
        cr_assert_eq(classes[r], r % 4);
}

Test(data, invalid_arguments)
{
    const auto classes = sampleClasses();
    cr_assert_throw(stratifiedSplit({}, {1.0}, 1), Error);
    cr_assert_throw(stratifiedSplit(classes, {}, 1), Error);
    cr_assert_throw(stratifiedSplit(classes, {0.5, 0.4}, 1), Error);
    cr_assert_throw(stratifiedSplit(classes, {1.2, -0.2}, 1), Error);
    cr_assert_throw(stratifiedSplit(classes, {1.0, 0.0}, 1), Error);
    cr_assert_throw(stratifiedKFold(classes, 1, 1), Error);
    cr_assert_throw(stratifiedKFold({0, 1}, 3, 1), Error);
    Dataset d{Matrix(3, 1), Matrix(3, 2)};
    const MatrixSource source(d);
    cr_assert_throw(SubsetSource(source, {0, 3}), Error);
    const SubsetSource subset(source, {2, 0});
    Matrix in(1, 1), out(1, 2);
    cr_assert_throw(subset.fill({2}, in, out), Error);
}
