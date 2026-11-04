// Measures the network file of ADR-0012: size, save time and load time, for two network sizes.
// Each timing is the median of 9 runs. The "raw" size is the lower bound of a binary format
// (8 bytes per number, no header), computed, not written.
//
// Usage: bench_serialization <output directory>   (see scripts/bench_serialization.sh)
// Writes: serialization.csv
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

constexpr int runs = 9;

Network build(const std::vector<std::size_t> &widths)
{
    Network net;
    for (std::size_t i = 0; i + 1 < widths.size(); i++) {
        net.add(std::make_unique<Dense>(widths[i], widths[i + 1], static_cast<std::uint32_t>(i)));
        net.add(i + 2 < widths.size() ? std::unique_ptr<Layer>(std::make_unique<ReLU>())
                                      : std::unique_ptr<Layer>(std::make_unique<Softmax>()));
    }
    net.info() = {"fen-onehot-768", widths.front(), {}, {}};
    for (std::size_t c = 0; c < widths.back(); c++)
        net.info().labels.push_back("class " + std::to_string(c));
    return net;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename F> double millis(F &&f)
{
    const auto start = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <output directory>\n";
        return 84;
    }
    const std::string dir = argv[1];
    const std::string tmp = dir + "/bench_serialization.nn";
    std::ofstream csv(dir + "/serialization.csv");
    csv << "network,parameters,file_bytes,raw_bytes,save_ms,load_ms\n";

    for (const auto &widths : std::vector<std::vector<std::size_t>>{{768, 128, 5},
                                                                    {768, 512, 256, 5}}) {
        Network net = build(widths);
        std::size_t parameters = 0;
        for (const Parameter &p : net.parameters())
            parameters += p.value->size();
        std::vector<double> save, load;
        for (int r = 0; r < runs; r++) {
            save.push_back(millis([&] { net.saveText(tmp); })); // the text format (ADR-0012)
            load.push_back(millis([&] { Network::load(tmp); }));
        }
        std::string name;
        for (std::size_t i = 0; i < widths.size(); i++)
            name += (i ? "-" : "") + std::to_string(widths[i]);
        csv << name << ',' << parameters << ',' << std::filesystem::file_size(tmp) << ','
            << parameters * 8 << ',' << median(save) << ',' << median(load) << '\n';
        std::remove(tmp.c_str());
    }
    return 0;
}
