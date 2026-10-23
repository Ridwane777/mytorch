#pragma once

#include <vector>

#include "my_torch/Loss.hpp"

namespace my_torch {

// Fonctions de coût. Toutes moyennent sur le batch (une ligne par exemple) et lèvent
// my_torch::Error si prediction et target n'ont pas la même forme.

// Erreur quadratique : moyenne sur le batch de Σ_j (p_j - t_j)². Pour la régression.
class MSE : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Erreur absolue : moyenne sur le batch de Σ_j |p_j - t_j|. Moins sensible aux valeurs aberrantes.
class MAE : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Quadratique sous delta, linéaire au-delà : MSE près de la cible, MAE loin d'elle.
class Huber : public Loss {
  public:
    explicit Huber(Scalar delta = 1.0);

    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;

  private:
    Scalar _delta;
};

// Entropie croisée binaire, sur des probabilités (sortie d'un Sigmoid) : moyenne sur le batch de
// -Σ_j [t_j·log p_j + (1 - t_j)·log(1 - p_j)]. Les probabilités sont bornées à [ε, 1 - ε].
class BinaryCrossEntropy : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Entropie croisée couplée à Softmax (ADR-0007). prediction = sortie du Softmax final, target =
// une distribution par ligne (one-hot). gradient() renvoie (p·W - w⊙q) / batch, soit
// (p - q) / batch sans pondération : le gradient par rapport à l'entrée du Softmax, sans passer
// par la division 1/p qui explose quand la bonne classe a une probabilité presque nulle.
//
// - labelSmoothing ∈ [0, 1[ : la cible devient q = (1 - s)·t + s / classes, ce qui empêche le
//   réseau de pousser les logits à l'infini.
// - classWeights : un poids par classe (vide = tous à 1), pour le déséquilibre des classes (#16).
//   La moyenne reste divisée par la taille du batch, pas par la somme des poids.
class CrossEntropy : public Loss {
  public:
    explicit CrossEntropy(Scalar labelSmoothing = 0, std::vector<Scalar> classWeights = {});

    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
    bool fusesSoftmax() const noexcept override;

  private:
    Matrix smoothed(const Matrix &prediction, const Matrix &target) const;
    Scalar weight(std::size_t cls) const noexcept;

    Scalar _smoothing;
    std::vector<Scalar> _weights;
};

} // namespace my_torch
