#include "my_torch/Matrix.hpp"

#include <algorithm>
#include <string>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

std::string shape(const Matrix &m)
{
    return std::to_string(m.rows()) + "x" + std::to_string(m.cols());
}

void requireSameShape(const char *op, const Matrix &a, const Matrix &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw Error(std::string(op) + ": " + shape(a) + " and " + shape(b));
}

} // namespace

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
        throw Error("matmul: " + shape(*this) + " by " + shape(other));
    Matrix result(_rows, other._cols);

    for (std::size_t i = 0; i < _rows; i++)
        for (std::size_t k = 0; k < _cols; k++) {
            const Scalar a = (*this)(i, k);
            for (std::size_t j = 0; j < other._cols; j++)
                result(i, j) += a * other(k, j);
        }
    return result;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    requireSameShape("operator+=", *this, other);
    for (std::size_t i = 0; i < _data.size(); i++)
        _data[i] += other._data[i];
    return *this;
}

Matrix Matrix::hadamard(const Matrix &other) const
{
    requireSameShape("hadamard", *this, other);
    Matrix result(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++)
        result._data[i] = _data[i] * other._data[i];
    return result;
}

Matrix Matrix::map(const std::function<Scalar(Scalar)> &fn) const
{
    Matrix result(_rows, _cols);

    for (std::size_t i = 0; i < _data.size(); i++)
        result._data[i] = fn(_data[i]);
    return result;
}

void Matrix::fill(Scalar value)
{
    std::fill(_data.begin(), _data.end(), value);
}

Matrix Matrix::addRowVector(const Matrix &row) const
{
    if (row._rows != 1 || row._cols != _cols)
        throw Error("addRowVector: " + shape(*this) + " and " + shape(row));
    Matrix result(*this);

    for (std::size_t r = 0; r < _rows; r++)
        for (std::size_t c = 0; c < _cols; c++)
            result(r, c) += row(0, c);
    return result;
}

Matrix Matrix::sumRows() const
{
    Matrix result(1, _cols);

    for (std::size_t r = 0; r < _rows; r++)
        for (std::size_t c = 0; c < _cols; c++)
            result(0, c) += (*this)(r, c);
    return result;
}

} // namespace my_torch
