#pragma once

#include <cstddef>
#include <vector>

#include "my_torch/Optimizer.hpp"

namespace my_torch {

// Optimiseurs : à chaque pas, ils lisent le gradient de chaque paramètre et déplacent sa valeur.
// Ceux qui ont une mémoire (vitesse, moyennes des gradients…) la rangent dans des matrices de la
// même forme que les paramètres, créées au premier appel et indexées par position : le même
// vecteur de paramètres, dans le même ordre, doit donc être donné à chaque pas, sinon
// my_torch::Error. Les gradients ne sont pas remis à zéro ici : c'est le rôle de la boucle
// d'entraînement. Tous lèvent my_torch::Error si un hyperparamètre est hors domaine.

// Descente de gradient : w -= lr·v, avec v = momentum·v + g.
// - momentum = 0 : SGD simple, w -= lr·g.
// - momentum > 0 : la vitesse v accumule les gradients passés, ce qui accélère le long des vallées
//   et amortit les oscillations.
// - nesterov : on avance avec g + momentum·v au lieu de v, c'est-à-dire en regardant un peu plus
// loin.
class SGD : public Optimizer {
  public:
    explicit SGD(Scalar learningRate, Scalar momentum = 0, bool nesterov = false);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _momentum;
    bool _nesterov;
    std::vector<Matrix> _velocity;
};

// Adam : un pas adapté à chaque poids. m est la moyenne mobile des gradients, v celle de leurs
// carrés (corrigées du biais de démarrage), et w -= lr·m̂ / (√v̂ + ε). Un poids au gradient
// toujours grand avance donc par pas d'environ lr, quel que soit l'échelle du gradient.
// - amsgrad : v̂ ne décroît jamais (maximum des v passés), ce qui garantit des pas qui ne
// regrossissent pas.
class Adam : public Optimizer {
  public:
    explicit Adam(Scalar learningRate = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999,
                  Scalar epsilon = 1e-8, bool amsgrad = false);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  protected:
    Adam(Scalar learningRate, Scalar weightDecay, Scalar beta1, Scalar beta2, Scalar epsilon,
         bool amsgrad, const char *name);

  private:
    Scalar _rate;
    Scalar _weightDecay;
    Scalar _beta1;
    Scalar _beta2;
    Scalar _epsilon;
    bool _amsgrad;
    const char *_name;
    std::size_t _steps = 0;
    std::vector<Matrix> _m;
    std::vector<Matrix> _v;
    std::vector<Matrix> _vmax;
};

// Adam avec décroissance des poids découplée : w -= lr·weightDecay·w, indépendamment du gradient.
// Une pénalité L2 ajoutée au gradient d'Adam serait « adaptée » comme lui et perdrait son effet.
class AdamW : public Adam {
  public:
    explicit AdamW(Scalar learningRate = 0.001, Scalar weightDecay = 0.01, Scalar beta1 = 0.9,
                   Scalar beta2 = 0.999, Scalar epsilon = 1e-8, bool amsgrad = false);
};

// RMSProp : divise le gradient par la racine d'une moyenne mobile de ses carrés, ce qui calme les
// poids aux gradients violents et accélère ceux aux gradients faibles.
class RMSProp : public Optimizer {
  public:
    explicit RMSProp(Scalar learningRate = 0.001, Scalar decay = 0.9, Scalar epsilon = 1e-8);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _decay;
    Scalar _epsilon;
    std::vector<Matrix> _square;
};

// Adagrad : divise par la racine de la somme de tous les carrés passés. Le pas de chaque poids ne
// fait que diminuer, ce qui convient aux problèmes convexes ou parcimonieux mais finit par figer
// l'apprentissage d'un réseau profond.
class Adagrad : public Optimizer {
  public:
    explicit Adagrad(Scalar learningRate = 0.01, Scalar epsilon = 1e-10);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _epsilon;
    std::vector<Matrix> _sum;
};

} // namespace my_torch
