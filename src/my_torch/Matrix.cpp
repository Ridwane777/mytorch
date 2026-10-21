#include "my_torch/Matrix.hpp"

#include <string>

#include "my_torch/Error.hpp"

namespace my_torch {

Matrix::Matrix(std::size_t rows, std::size_t cols, Scalar value)
    : _rows(rows), _cols(cols), _data(rows * cols, value)
{
}

std::size_t Matrix::rows() const noexcept
{
    return _rows;
}

std::size_t Matrix::cols() const noexcept
{
    return _cols;
}

std::size_t Matrix::size() const noexcept
{
    return _data.size();
}

Scalar &Matrix::operator()(std::size_t row, std::size_t col)
{
    return _data[row * _cols + col];
}

Scalar Matrix::operator()(std::size_t row, std::size_t col) const
{
    return _data[row * _cols + col];
}

Scalar *Matrix::data() noexcept
{
    return _data.data();
}

const Scalar *Matrix::data() const noexcept
{
    return _data.data();
}

Matrix Matrix::transpose() const
{
    Matrix result(_cols, _rows);

    for (std::size_t r = 0; r < _rows; r++)
        for (std::size_t c = 0; c < _cols; c++)
            result(c, r) = (*this)(r, c);
    return result;
}

// Version naïve, ordre i-k-j pour parcourir les lignes en mémoire contiguë.
// L'optimisation (blocs, SIMD) est l'objet de #33.
Matrix Matrix::matmul(const Matrix &other) const
{
    if (_cols != other._rows)
        throw Error("matmul: " + std::to_string(_rows) + "x" + std::to_string(_cols) + " by " +
                    std::to_string(other._rows) + "x" + std::to_string(other._cols));
    Matrix result(_rows, other._cols);

    for (std::size_t i = 0; i < _rows; i++)
        for (std::size_t k = 0; k < _cols; k++) {
            const Scalar a = (*this)(i, k);
            for (std::size_t j = 0; j < other._cols; j++)
                result(i, j) += a * other(k, j);
        }
    return result;
}

} // namespace my_torch
