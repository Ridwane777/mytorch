#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace my_torch {

// double rather than float: the finite-difference gradient check (#3) requires
// a relative error < 1e-6, out of reach in single precision. See ADR-0003.
using Scalar = double;

// Dense matrix, contiguous row-major storage. A batch is a matrix
// (batch size × number of features): one row per example.
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
    // Matrix product; throws my_torch::Error if the dimensions do not match. Blocked for the cache
    // and spread over the threads of Parallel.hpp (ADR-0024); every output element is summed in
    // the same order as the plain triple loop, so the result is the same to the bit whatever the
    // number of threads.
    Matrix matmul(const Matrix &other) const;
    // thisᵀ · other and this · otherᵀ without building the transpose, with the same results to the
    // bit as transpose().matmul(other) and matmul(other.transpose()).
    Matrix transposeMatmul(const Matrix &other) const;
    Matrix matmulTranspose(const Matrix &other) const;

    // Element-wise operations; throw my_torch::Error if the shapes differ.
    Matrix &operator+=(const Matrix &other);
    Matrix hadamard(const Matrix &other) const;
    Matrix map(const std::function<Scalar(Scalar)> &fn) const;
    void fill(Scalar value);

    // Adds the row vector (1 × cols) to every row: the bias shared by the whole batch.
    Matrix addRowVector(const Matrix &row) const;
    // Sum of all rows (1 × cols): the bias gradient over the batch.
    Matrix sumRows() const;

  private:
    std::size_t _rows = 0;
    std::size_t _cols = 0;
    std::vector<Scalar> _data;
};

} // namespace my_torch
