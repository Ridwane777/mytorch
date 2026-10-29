// Time of Network::fit on the final architecture (901 -> 256 -> 128 -> 5, Adam, batches of 64) on
// synthetic data, with or without a checkpoint (#18). Built by scripts/bench_report.sh against the
// current library and against a commit before the non-finite checks of ADR-0031, to measure their
// cost; the same source compiles with both (no checkpoint argument is passed when CHECKPOINT is not
// defined).
// Usage: bench_overheads EPOCHS [CHECKPOINT_FILE]; prints the seconds of fit.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "my_torch/my_torch.hpp"

using namespace my_torch;

int main(int argc, char **argv)
{
    if (argc < 2)
        return 84;
    Matrix x(8192, 901);
    Matrix t(8192, 5);
    for (std::size_t i = 0; i < x.size(); i++)
        x.data()[i] = static_cast<Scalar>(i * 2654435761u % 1000) / 1000;
    for (std::size_t i = 0; i < 8192; i++)
        t(i, i % 5) = 1;
    Network net;
    net.add(std::make_unique<Dense>(901, 256, 1));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dense>(256, 128, 2));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Dense>(128, 5, 3));
    net.add(std::make_unique<Softmax>());
    CrossEntropy loss;
    Adam adam(0.001);
    TrainConfig config;
    config.epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    config.batchSize = 64;
    const auto start = std::chrono::steady_clock::now();
#ifdef CHECKPOINT
    net.fit({x, t}, nullptr, loss, adam, config, nullptr, argc > 2 ? argv[2] : "");
#else
    net.fit({x, t}, nullptr, loss, adam, config);
#endif
    std::printf("%.4f\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return 0;
}
