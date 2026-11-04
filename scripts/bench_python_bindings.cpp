// C++ side of the Python bindings benchmark (ADR-0044, docs/performance/25-python-bindings.md):
// times Network::predict on encoded rows (pytorch_reference export: one byte per value) and writes
// the probabilities as raw doubles, for bonus/python/bench.py to compare with the module's.
// Usage: bench_python_bindings NETWORK X.u8 ROWS REPEATS OUT.f64
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "my_torch/Error.hpp"
#include "my_torch/Network.hpp"

using my_torch::Error;
using my_torch::Matrix;
using my_torch::Network;

int main(int argc, char **argv)
{
    try {
        if (argc != 6)
            throw Error("usage: bench_python_bindings NETWORK X.u8 ROWS REPEATS OUT.f64");
        Network network = Network::load(argv[1]);
        const std::size_t rows = std::stoul(argv[3]);
        const std::size_t repeats = std::stoul(argv[4]);
        const std::size_t cols = network.info().inputs;

        std::ifstream in(argv[2], std::ios::binary);
        std::vector<unsigned char> bytes(rows * cols);
        if (!in.read(reinterpret_cast<char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())))
            throw Error(std::string("cannot read ") + argv[2]);
        Matrix inputs(rows, cols);
        std::copy(bytes.begin(), bytes.end(), inputs.data());

        Matrix out;
        for (std::size_t r = 0; r < repeats; r++) {
            const auto start = std::chrono::steady_clock::now();
            out = network.predict(inputs);
            const std::chrono::duration<double> s = std::chrono::steady_clock::now() - start;
            std::cout << s.count() << '\n';
        }
        std::ofstream file(argv[5], std::ios::binary);
        file.write(reinterpret_cast<const char *>(out.data()),
                   static_cast<std::streamsize>(out.rows() * out.cols() * sizeof(double)));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 84;
    }
}
