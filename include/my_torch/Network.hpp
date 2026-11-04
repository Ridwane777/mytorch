#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
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

struct EpochStats;

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
    // Gradient clipping (ADR-0031), applied between backward and the optimizer step; 0 = off.
    // clipValue clamps each gradient to [-clipValue, clipValue]; clipNorm then rescales all the
    // gradients together so that their global L2 norm is at most clipNorm, keeping the direction.
    Scalar clipNorm = 0;
    Scalar clipValue = 0;
    // Called by fit at the end of each epoch it runs, with the 0-based epoch and its measurements
    // (ADR-0035), before early stopping decides; the parameters then hold the weights of that
    // epoch and the gradients of its last batch. Not saved in network files. Empty = none.
    std::function<void(std::size_t epoch, const EpochStats &stats)> onEpoch;
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

// Monte Carlo dropout (ADR-0034): the mean of several predictions with the dropout layers active,
// and three measures of how much they disagree, per row (natural logarithms):
// - entropy: of the mean probabilities, the total uncertainty;
// - mutualInformation: entropy minus the mean entropy of each pass, the part due to the passes
//   disagreeing (the uncertainty of the model rather than of the position);
// - disagreement: share of the passes whose top class is not the most frequent one.
struct UncertainPrediction {
    Matrix mean;
    std::vector<Scalar> entropy;
    std::vector<Scalar> mutualInformation;
    std::vector<Scalar> disagreement;
};

class Network;

// Table of the layers (ADR-0035), as Keras' summary(): index, type, output width and number of
// trained parameters per layer, then the total. Widths follow Layer::outputWidth from the input
// width of the network's info.
std::string modelSummary(Network &network);

// Sequential network: layers apply in the order they were added. A graph architecture
// (residual blocks, #31) will get its own ADR.
class Network {
  public:
    void add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix &input, bool training = false);
    Matrix predict(const Matrix &input);
    // `passes` predictions with only the Dropout layers in training mode (BatchNorm and the others
    // stay in prediction mode), their generators seeded from `seed`: the same seed gives the same
    // result, and the network's own generators are left as they were. The last layer must output
    // probabilities (softmax). Throws my_torch::Error if passes is 0 or the network has no dropout
    // layer, since every pass would then be the same.
    // d probability(output) / d input, one row per input row, through every layer in prediction
    // mode (saliency, ADR-0035). The last layer must output probabilities (softmax). The
    // parameters' gradients are left as they were. Throws my_torch::Error on an output out of range.
    Matrix inputGradient(const Matrix &input, std::size_t output);
    UncertainPrediction predictMonteCarlo(const Matrix &input, std::size_t passes,
                                          std::uint64_t seed = 1);

    // Mini-batch training (ADR-0009). Each epoch: apply the scheduler if any, shuffle with the
    // seed, then for each batch zero the gradients, forward, loss, backward, optimizer step. When
    // loss.fusesSoftmax(), the last layer must be a Softmax and its backward is skipped (ADR-0007).
    // Throws my_torch::Error on empty or mismatched data, a zero batch size, a negative clipping
    // threshold or a missing Softmax. Stops with my_torch::Error on the first non-finite loss,
    // gradient or weight (ADR-0031), naming the epoch, the batch and the layer (counted from 1).
    // With a checkpoint path (ADR-0032), the whole training state is saved there at the end of
    // every epoch, and when the file already exists training resumes after its last epoch: the
    // returned history then includes the epochs run before the interruption. The checkpoint must
    // come from the same settings (except epochs), layers, optimizer and data sizes.
    History fit(const DataSource &train, const DataSource *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config,
                const Scheduler *scheduler = nullptr, const std::string &checkpoint = {});
    // Same, on matrices held in memory (wrapped in MatrixSource).
    History fit(const Dataset &train, const Dataset *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config,
                const Scheduler *scheduler = nullptr, const std::string &checkpoint = {});

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
