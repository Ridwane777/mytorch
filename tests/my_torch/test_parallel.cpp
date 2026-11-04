#include <criterion/criterion.h>

#include <atomic>
#include <memory>
#include <vector>

#include "gradient_check.hpp"

using namespace my_torch;

namespace {

// The plain triple loop the library used before ADR-0024: the reference to match to the bit.
Matrix naive(const Matrix &a, const Matrix &b)
{
    Matrix r(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); i++)
        for (std::size_t k = 0; k < a.cols(); k++) {
            const Scalar x = a(i, k);
            for (std::size_t j = 0; j < b.cols(); j++)
                r(i, j) += x * b(k, j);
        }
    return r;
}

bool same(const Matrix &a, const Matrix &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        return false;
    for (std::size_t i = 0; i < a.size(); i++)
        if (a.data()[i] != b.data()[i])
            return false;
    return true;
}

struct Shape {
    std::size_t n, k, m;
};

// Sizes below and above the blocks and the parallel threshold, and the shapes of the network.
const std::vector<Shape> shapes = {{1, 1, 1},      {7, 13, 5},     {3, 65, 257},
                                   {64, 901, 256}, {64, 256, 128}, {130, 70, 300}};

} // namespace

Test(parallel, blocked_product_equals_the_plain_loop_bit_for_bit)
{
    for (std::size_t t : {1, 3, 12}) {
        setThreadCount(t);
        for (const Shape &s : shapes) {
            const Matrix a = gradient_check::random(s.n, s.k, 1);
            const Matrix b = gradient_check::random(s.k, s.m, 2);
            cr_assert(same(a.matmul(b), naive(a, b)), "%zu threads, %zux%zux%zu", t, s.n, s.k, s.m);
        }
    }
    setThreadCount(0);
}

Test(parallel, transpose_products_equal_the_copied_transpose_bit_for_bit)
{
    for (std::size_t t : {1, 5}) {
        setThreadCount(t);
        for (const Shape &s : shapes) {
            const Matrix x = gradient_check::random(s.n, s.k, 3);
            const Matrix g = gradient_check::random(s.n, s.m, 4);
            const Matrix w = gradient_check::random(s.k, s.m, 5);
            cr_assert(same(x.transposeMatmul(g), x.transpose().matmul(g)));
            cr_assert(same(g.matmulTranspose(w), g.matmul(w.transpose())));
        }
    }
    setThreadCount(0);
}

Test(parallel, products_reject_mismatched_shapes)
{
    cr_assert_throw(Matrix(2, 3).transposeMatmul(Matrix(3, 2)), Error);
    cr_assert_throw(Matrix(2, 3).matmulTranspose(Matrix(2, 4)), Error);
}

Test(parallel, parallel_for_covers_every_index_once)
{
    for (std::size_t t : {1, 2, 7, 16}) {
        setThreadCount(t);
        for (std::size_t count : {0u, 1u, 5u, 1000u}) {
            std::vector<std::atomic<int>> hits(count);
            parallelFor(count, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++)
                    hits[i]++;
            });
            for (std::size_t i = 0; i < count; i++)
                cr_assert_eq(hits[i].load(), 1, "%zu threads, index %zu of %zu", t, i, count);
        }
    }
    setThreadCount(0);
    cr_assert_geq(threadCount(), 1);
}

Test(parallel, training_is_identical_with_one_or_many_threads)
{
    const auto train = [](std::size_t threads) {
        setThreadCount(threads);
        Network net;
        net.add(std::make_unique<Dense>(40, 300, 1));
        net.add(std::make_unique<ReLU>());
        net.add(std::make_unique<Dense>(300, 3, 2));
        net.add(std::make_unique<Softmax>());
        Matrix x = gradient_check::random(256, 40, 7);
        Matrix t(256, 3);
        for (std::size_t r = 0; r < 256; r++)
            t(r, r % 3) = 1;
        CrossEntropy loss;
        Adam adam(0.01);
        TrainConfig config;
        config.epochs = 3;
        config.batchSize = 64;
        net.fit({x, t}, nullptr, loss, adam, config);
        std::vector<Scalar> w;
        for (const Parameter &p : net.parameters())
            w.insert(w.end(), p.value->data(), p.value->data() + p.value->size());
        return w;
    };
    const auto one = train(1);
    cr_assert(one == train(12), "12 threads trained other weights");
    setThreadCount(0);
}
