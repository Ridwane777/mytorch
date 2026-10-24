#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "my_torch/Layer.hpp"
#include "my_torch/Loss.hpp"
#include "my_torch/Optimizer.hpp"
#include "my_torch/Schedulers.hpp"

namespace my_torch {

// Inputs and expected outputs, one row per example.
struct Dataset {
    Matrix inputs;
    Matrix targets;
};

struct TrainConfig {
    std::size_t epochs = 10;
    std::size_t batchSize = 32;
    std::uint64_t seed = 42;
    bool shuffle = true;
};

// Measurements of one epoch; the detailed metrics (#11) will be added here.
struct EpochStats {
    Scalar trainLoss = 0;
    Scalar validationLoss = 0;
};

using History = std::vector<EpochStats>;

// Sequential network: layers apply in the order they were added. A graph architecture
// (residual blocks, #31) will get its own ADR.
class Network {
  public:
    void add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix &input, bool training = false);
    Matrix predict(const Matrix &input);

    // Mini-batch training (ADR-0009). Each epoch: apply the scheduler if any, shuffle with the
    // seed, then for each batch zero the gradients, forward, loss, backward, optimizer step. When
    // loss.fusesSoftmax(), the last layer must be a Softmax and its backward is skipped (ADR-0007).
    // Throws my_torch::Error on empty or mismatched data, a zero batch size or a missing Softmax.
    History fit(const Dataset &train, const Dataset *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config,
                const Scheduler *scheduler = nullptr);

    std::vector<Parameter> parameters();
    const std::vector<std::unique_ptr<Layer>> &layers() const noexcept;

    // Format defined by the ADR of #8. Throw my_torch::Error if the file is unreadable.
    void save(const std::string &path) const;
    static Network load(const std::string &path);

  private:
    std::vector<std::unique_ptr<Layer>> _layers;
};

} // namespace my_torch
