#pragma once

#include <cstddef>
#include <vector>

namespace my_torch {

// double plutôt que float : la vérification des gradients par différences finies (#3) exige
// une erreur relative < 1e-6, hors de portée en simple précision. Voir ADR-0003.
using Scalar = double;

// Matrice dense, stockage contigu en ligne (row-major). Un batch est une matrice
// (taille du batch × nombre de caractéristiques) : une ligne par exemple.
class Matrix {
  public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, Scalar value = 0);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t size() const noexcept;

    Scalar &operator()(std::size_t row, std::size_t col);
    Scalar operator()(std::size_t row, std::size_t col) const;
    Scalar *data() noexcept;
    const Scalar *data() const noexcept;

    Matrix transpose() const;
    // Produit matriciel ; lève my_torch::Error si les dimensions ne correspondent pas.
    Matrix matmul(const Matrix &other) const;

  private:
    std::size_t _rows = 0;
    std::size_t _cols = 0;
    std::vector<Scalar> _data;
};

} // namespace my_torch
