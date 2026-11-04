// Text format (ADR-0012) against binary format (ADR-0042) for network files: size, save time and
// load time of the same network in both formats, the median of 9 runs each.
//
// Usage: bench_binary_format measure OUT_CSV NETWORK…   one CSV line per network and format (see
//                                                        scripts/bench_binary_format.sh)
//        bench_binary_format convert IN OUT              loads IN (either format), saves OUT in
//                                                        the binary format
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

namespace {

constexpr int runs = 9;

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
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "convert" && argc == 4) {
            Network::load(argv[2]).save(argv[3]);
            return 0;
        }
        if (mode != "measure" || argc < 4) {
            std::cerr << "usage: " << argv[0] << " measure OUT_CSV NETWORK… | convert IN OUT\n";
            return 84;
        }
        std::ofstream csv(argv[2]);
        csv << "network,parameters,format,file_bytes,save_ms,load_ms\n";
        const std::string tmp = std::string(argv[2]) + ".tmp";
        for (int a = 3; a < argc; a++) {
            Network net = Network::load(argv[a]);
            std::size_t parameters = 0;
            for (const Parameter &p : net.parameters())
                parameters += p.value->size();
            for (const std::string format : {"text", "binary"}) {
                std::vector<double> save, load;
                for (int r = 0; r < runs; r++) {
                    save.push_back(millis([&] {
                        if (format == "text")
                            net.saveText(tmp);
                        else
                            net.save(tmp);
                    }));
                    load.push_back(millis([&] { Network::load(tmp); }));
                }
                csv << std::filesystem::path(argv[a]).filename().string() << ',' << parameters
                    << ',' << format << ',' << std::filesystem::file_size(tmp) << ','
                    << median(save) << ',' << median(load) << '\n';
            }
        }
        std::remove(tmp.c_str());
    } catch (const std::exception &error) {
        std::cerr << "bench_binary_format: " << error.what() << '\n';
        return 84;
    }
    return 0;
}
