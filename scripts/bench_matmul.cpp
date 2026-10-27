// Speed of the matrix products of the final network (docs/performance/12-matmul.md, ADR-0024): the
// plain triple loop used before ADR-0024, then the blocked product on 1 to N threads, for the
// three products of a Dense 901 -> 256 layer on a batch of 64. Median of 7 timings, each repeating
// the product for at least 0.2 s. Writes matmul.csv to the given directory.
// Usage: bench_matmul OUTPUT_DIRECTORY THREAD_COUNTS... (e.g. 1 2 4 8 12)
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

Matrix filled(std::size_t rows, std::size_t cols, Scalar seed)
{
    Matrix m(rows, cols);
    for (std::size_t i = 0; i < m.size(); i++)
        m.data()[i] = static_cast<Scalar>((i * 2654435761u) % 1000) / 1000.0 - 0.5 + seed;
    return m;
}

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

double seconds(const std::function<void()> &f)
{
    std::vector<double> times;
    for (int t = 0; t < 7; t++) {
        int repeats = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed = 0;
        do {
            f();
            repeats++;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } while (elapsed < 0.2);
        times.push_back(elapsed / repeats);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " OUTPUT_DIRECTORY THREAD_COUNTS...\n";
        return 84;
    }
    const Matrix x = filled(64, 901, 0);   // a batch of 64 positions
    const Matrix w = filled(901, 256, 0.1); // the first Dense layer
    const Matrix g = filled(64, 256, 0.2);  // the gradient of its output
    struct Product {
        const char *name;
        std::function<Matrix()> blocked;
        std::function<Matrix()> plain;
        double flops;
    };
    const double f = 2.0 * 64 * 901 * 256;
    const std::vector<Product> products = {
        {"forward X·W", [&] { return x.matmul(w); }, [&] { return naive(x, w); }, f},
        {"weight gradient Xᵀ·G", [&] { return x.transposeMatmul(g); },
         [&] { return naive(x.transpose(), g); }, f},
        {"input gradient G·Wᵀ", [&] { return g.matmulTranspose(w); },
         [&] { return naive(g, w.transpose()); }, f},
    };
    std::ofstream csv(std::string(argv[1]) + "/matmul.csv");
    csv << "product,version,threads,seconds,gflops\n";
    for (const Product &p : products) {
        setThreadCount(1);
        const double plain = seconds([&] { (void)p.plain(); });
        csv << p.name << ",plain loop,1," << plain << ',' << p.flops / plain / 1e9 << '\n';
        for (int a = 2; a < argc; a++) {
            const std::size_t threads = std::stoul(argv[a]);
            setThreadCount(threads);
            const double s = seconds([&] { (void)p.blocked(); });
            csv << p.name << ",blocked," << threads << ',' << s << ',' << p.flops / s / 1e9 << '\n';
        }
    }
    return 0;
}
