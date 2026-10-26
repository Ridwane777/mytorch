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

// A loss, optimizer or scheduler described by its type and its settings, as written in the network
// file: "adam lr=0.001 beta1=0.9" ↔ {"adam", {{"lr", "0.001"}, {"beta1", "0.9"}}}. The objects are
// built from it by Factory.hpp (ADR-0012).
struct ComponentSpec {
    std::string type;
    std::vector<std::pair<std::string, std::string>> params;
};

// Inputs and expected outputs, one row per example.
struct Dataset {
    Matrix inputs;
    Matrix targets;
};

// Where fit takes its examples (ADR-0021). fit asks for one batch at a time, so a source can build
// its rows on demand (the analyzer encodes chess positions batch by batch) instead of holding every
// example as a matrix in memory.
class DataSource {
  public:
    virtual ~DataSource() = default;

    virtual std::size_t size() const = 0;
    virtual std::size_t inputWidth() const = 0;
    virtual std::size_t targetWidth() const = 0;
    // Writes example rows[i] into row i of inputs and targets, which fit has sized to
    // rows.size() × inputWidth() and rows.size() × targetWidth() and filled with zeros.
    virtual void fill(const std::vector<std::size_t> &rows, Matrix &inputs,
                      Matrix &targets) const = 0;
};

// A Dataset seen as a DataSource: rows are copied from the matrices.
class MatrixSource : public DataSource {
  public:
    // Keeps a reference: data must outlive the source. Throws my_torch::Error if the inputs and
    // targets have different row counts.
    explicit MatrixSource(const Dataset &data);

    std::size_t size() const override;
    std::size_t inputWidth() const override;
    std::size_t targetWidth() const override;
    void fill(const std::vector<std::size_t> &rows, Matrix &inputs, Matrix &targets) const override;

  private:
    const Dataset &_data;
};

// Validation is evaluated this many examples at a time, so it never needs one big matrix.
constexpr std::size_t EVALUATION_CHUNK = 4096;

struct TrainConfig {
    std::size_t epochs = 10;
    std::size_t batchSize = 32;
    std::uint64_t seed = 42;
    bool shuffle = true;
    // Early stopping (ADR-0010), off when patience is 0. Training stops after `patience` epochs
    // whose validation loss did not beat the best one by more than minDelta; with restoreBest,
    // the weights of the best epoch are put back at the end. Needs a validation set.
    std::size_t patience = 0;
    Scalar minDelta = 0;
    bool restoreBest = true;
    // Fraction of a single data file held out for validation by the analyzer's --train
    // (ADR-0014), in [0, 1[. fit itself takes the validation set as an argument.
    Scalar validation = 0;
    // Alternative to validation (ADR-0022): the last validationTail lines of the file are the
    // validation set, taken in order, never drawn at random. Lets a pipeline append an untouched
    // validation set after an augmented training set without any leak. 0 = unused; giving both
    // validation and validationTail is an error.
    std::size_t validationTail = 0;
};

// The numbers 0..count-1 in an order shuffled by seed, with the same portable Fisher-Yates as
// fit (ADR-0009): the same seed gives the same order on every toolchain.
std::vector<std::size_t> permutation(std::size_t count, std::uint64_t seed);

// Measurements of one epoch; the detailed metrics (#11) will be added here.
struct EpochStats {
    Scalar trainLoss = 0;      // data loss only, comparable with validationLoss
    Scalar validationLoss = 0; // data loss on the validation set, 0 without one
    Scalar penalty = 0;        // sum of the layers' penalty(), averaged over the epoch's batches
    // Fraction of examples whose predicted class (argmax, or >= 0.5 for one output) matches the
    // target's; meaningful for classification only (ADR-0011). Training accuracy is measured on
    // the batches as they are trained, so with dropout it is the accuracy of the dropped network.
    Scalar trainAccuracy = 0;
    Scalar validationAccuracy = 0; // 0 without a validation set
};

using History = std::vector<EpochStats>;

// Everything --train needs besides the layers.
struct TrainingSpec {
    ComponentSpec loss{"cross_entropy", {}};
    ComponentSpec optimizer{"adam", {}};
    ComponentSpec scheduler{"none", {}};
    TrainConfig config;
};

// What the analyzer needs to use the network, stored in the network file with the layers.
struct ModelInfo {
    std::string encoding;            // name of the input encoding (#14), one word
    std::size_t inputs = 0;          // width of an input row
    std::vector<std::string> labels; // name of each output, in order
    TrainingSpec training;
};

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
    History fit(const DataSource &train, const DataSource *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config,
                const Scheduler *scheduler = nullptr);
    // Same, on matrices held in memory (wrapped in MatrixSource).
    History fit(const Dataset &train, const Dataset *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config,
                const Scheduler *scheduler = nullptr);

    std::vector<Parameter> parameters();
    const std::vector<std::unique_ptr<Layer>> &layers() const noexcept;

    ModelInfo &info() noexcept;
    const ModelInfo &info() const noexcept;

    // Versioned text file with a checksum (ADR-0012). Both check that the info and the layers are
    // consistent (input width, layer widths, one output per label) and throw my_torch::Error
    // otherwise; load also reports the line of a malformed or corrupted file.
    void save(const std::string &path) const;
    static Network load(const std::string &path);
    // Builds an untrained network from a configuration file (ADR-0013): the network file syntax
    // without weights or checksum, with comments, optional sections and "init seed=…". Weights
    // are drawn from seeds derived from that seed plus index, so each index gives a different
    // network and the same index always gives the same one.
    static Network fromConfig(const std::string &path, std::uint64_t index = 0);

  private:
    static Network parse(const std::string &path, bool config, std::uint64_t index);

    std::vector<std::unique_ptr<Layer>> _layers;
    ModelInfo _info;
};

} // namespace my_torch
