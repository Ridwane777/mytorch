#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

// Training checkpoints (ADR-0032). fit writes one at the end of every epoch when given a path, and
// resumes from it when the file exists, so that an interrupted training continued from its last
// checkpoint ends with the same weights as an uninterrupted one.

// Matrices in a state block, written with 17 significant digits so that every double reads back
// exactly. Used by Optimizer::saveState / loadState and Layer::saveState / loadState.
void writeMatrices(std::ostream &out, const std::vector<Matrix> &matrices);
// Throws my_torch::Error on malformed text.
std::vector<Matrix> readMatrices(std::istream &in);

// Everything fit needs to continue a training after `epoch` finished epochs.
struct TrainingState {
    std::size_t epoch = 0;
    std::string shuffle; // text state of the shuffling generator
    History history;
    // Early stopping (ADR-0010).
    Scalar best = 0;
    std::size_t sinceBest = 0;
    std::vector<Matrix> bestWeights;
    std::vector<std::string> bestLayers; // Layer::saveState at the best epoch (BatchNorm averages)
    std::vector<Matrix> weights;
    std::string optimizer;           // Optimizer::saveState
    std::vector<std::string> layers; // Layer::saveState, one per layer
};

// Writes the state to path through a temporary file renamed at the end, so that a crash while
// writing leaves the previous checkpoint intact. fingerprint identifies the training (its settings,
// its layers, the size of its data) and is checked on loading.
void saveCheckpoint(const std::string &path, std::uint64_t fingerprint, const TrainingState &state);
// std::nullopt when the file does not exist. Throws my_torch::Error on a corrupted file (checksum)
// or one written by another training (fingerprint).
std::optional<TrainingState> loadCheckpoint(const std::string &path, std::uint64_t fingerprint);

// FNV-1a 64 of a text, for fingerprints.
std::uint64_t fingerprintOf(const std::string &text);

} // namespace my_torch
