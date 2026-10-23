#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "my_torch/Matrix.hpp"

namespace my_torch {

// Planificateurs du taux d'apprentissage : une fonction pure de l'époque (la première est 0).
// La boucle d'entraînement l'applique avec optimizer.setLearningRate(scheduler.rate(epoch)) au
// début de chaque époque. Sans état, un planificateur donne la même valeur pour la même époque,
// ce qui permet de reprendre un entraînement (#35) sans rien sauvegarder. Un taux élevé au début
// avance vite ; baisser ensuite évite d'osciller autour du minimum.
class Scheduler {
  public:
    virtual ~Scheduler() = default;

    virtual Scalar rate(std::size_t epoch) const = 0;
    virtual std::string type() const = 0;
};

// Toujours le même taux.
class ConstantRate : public Scheduler {
  public:
    explicit ConstantRate(Scalar base);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
};

// Multiplie le taux par gamma toutes les stepSize époques : base·gamma^⌊époque / stepSize⌋.
class StepDecay : public Scheduler {
  public:
    StepDecay(Scalar base, std::size_t stepSize, Scalar gamma = 0.1);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    std::size_t _stepSize;
    Scalar _gamma;
};

// Décroissance continue : base·gamma^époque.
class ExponentialDecay : public Scheduler {
  public:
    ExponentialDecay(Scalar base, Scalar gamma);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    Scalar _gamma;
};

// Descend de base à minimum selon une demi-cosinusoïde sur totalEpochs époques :
// minimum + (base - minimum)·(1 + cos(π·époque / totalEpochs)) / 2, puis reste à minimum.
// Lente au début et à la fin, rapide au milieu.
class CosineAnnealing : public Scheduler {
  public:
    CosineAnnealing(Scalar base, std::size_t totalEpochs, Scalar minimum = 0);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    std::size_t _totalEpochs;
    Scalar _minimum;
};

// Échauffement : monte linéairement jusqu'au taux de départ de `inner` pendant warmupEpochs
// époques (base·(époque + 1) / warmupEpochs), puis laisse `inner` prendre le relais en repartant
// de son époque 0. Évite les premiers pas trop grands d'Adam, dont les moyennes sont encore vides.
class Warmup : public Scheduler {
  public:
    Warmup(std::unique_ptr<Scheduler> inner, std::size_t warmupEpochs);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    std::unique_ptr<Scheduler> _inner;
    std::size_t _warmupEpochs;
};

} // namespace my_torch
