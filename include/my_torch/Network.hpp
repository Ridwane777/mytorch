#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "my_torch/Layer.hpp"
#include "my_torch/Loss.hpp"
#include "my_torch/Optimizer.hpp"

namespace my_torch {

// Entrées et sorties attendues, une ligne par exemple.
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

// Mesures d'une époque ; les métriques détaillées (#11) s'ajouteront ici.
struct EpochStats {
    Scalar trainLoss = 0;
    Scalar validationLoss = 0;
};

using History = std::vector<EpochStats>;

// Réseau séquentiel : les couches s'appliquent dans l'ordre d'ajout. Une architecture en
// graphe (blocs résiduels, #31) fera l'objet d'un ADR séparé.
class Network {
  public:
    void add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix &input, bool training = false);
    Matrix predict(const Matrix &input);

    History fit(const Dataset &train, const Dataset *validation, Loss &loss,
                Optimizer &optimizer, const TrainConfig &config);

    std::vector<Parameter> parameters();
    const std::vector<std::unique_ptr<Layer>> &layers() const noexcept;

    // Format défini par l'ADR de #8. Lèvent my_torch::Error si le fichier est illisible.
    void save(const std::string &path) const;
    static Network load(const std::string &path);

  private:
    std::vector<std::unique_ptr<Layer>> _layers;
};

} // namespace my_torch
