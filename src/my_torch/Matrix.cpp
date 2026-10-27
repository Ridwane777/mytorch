#include "my_torch/Matrix.hpp"

#include <algorithm>
#include <string>

#include "my_torch/Error.hpp"
#include "my_torch/Parallel.hpp"

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

// Below this many multiply-adds a product stays on one thread: splitting it costs more than it saves.
constexpr std::size_t parallelThreshold = std::size_t{1} << 16;
// Block sizes: a block of B (kBlock × jBlock doubles, 128 KiB) stays in the L2 cache while every
// row of A uses it.
constexpr std::size_t kBlock = 64;
constexpr std::size_t jBlock = 256;

namespace {

// result(i, j) += Σ_k A(i, k) · B(k, j), rows [rowBegin, rowEnd), with A(i, k) read at
// a[i * aRow + k * aCol] and B row-major. For each element the products are added in increasing k,
// as in the plain i-k-j loop: blocking over k and j does not change that order.
void productRows(const Scalar *a, std::size_t aRow, std::size_t aCol, const Scalar *b,
                 Scalar *result, std::size_t inner, std::size_t cols, std::size_t rowBegin,
                 std::size_t rowEnd)
{
    for (std::size_t j0 = 0; j0 < cols; j0 += jBlock) {
        const std::size_t j1 = std::min(j0 + jBlock, cols);
        for (std::size_t k0 = 0; k0 < inner; k0 += kBlock) {
            const std::size_t k1 = std::min(k0 + kBlock, inner);
            for (std::size_t i = rowBegin; i < rowEnd; i++) {
                Scalar *out = result + i * cols;
                for (std::size_t k = k0; k < k1; k++) {
                    const Scalar value = a[i * aRow + k * aCol];
                    const Scalar *row = b + k * cols;
                    for (std::size_t j = j0; j < j1; j++)
                        out[j] += value * row[j];
                }
            }
        }
    }
}

void forRows(std::size_t rows, std::size_t work,
             const std::function<void(std::size_t, std::size_t)> &body)
{
    if (work < parallelThreshold)
        body(0, rows);
    else
        parallelFor(rows, body);
}

} // namespace

Matrix Matrix::matmul(const Matrix &other) const
{
    if (_cols != other._rows)
        throw Error("matmul: " + shape(*this) + " by " + shape(other));
    Matrix result(_rows, other._cols);

    forRows(_rows, _rows * _cols * other._cols, [&](std::size_t begin, std::size_t end) {
        productRows(data(), _cols, 1, other.data(), result.data(), _cols, other._cols, begin, end);
    });
    return result;
}

Matrix Matrix::transposeMatmul(const Matrix &other) const
{
    if (_rows != other._rows)
        throw Error("transposeMatmul: " + shape(*this) + " by " + shape(other));
    Matrix result(_cols, other._cols);

    // Row i of the result reads column i of this: A(i, k) = this(k, i).
    forRows(_cols, _rows * _cols * other._cols, [&](std::size_t begin, std::size_t end) {
        productRows(data(), 1, _cols, other.data(), result.data(), _rows, other._cols, begin, end);
    });
    return result;
}

Matrix Matrix::matmulTranspose(const Matrix &other) const
{
    if (_cols != other._cols)
        throw Error("matmulTranspose: " + shape(*this) + " by " + shape(other));
    Matrix result(_rows, other._rows);

    // result(i, j) is the dot product of row i of this and row j of other, both contiguous; the
    // products are added in increasing k from 0, as matmul(other.transpose()) does.
    forRows(_rows, _rows * _cols * other._rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            const Scalar *left = data() + i * _cols;
            for (std::size_t j = 0; j < other._rows; j++) {
                const Scalar *right = other.data() + j * _cols;
                Scalar sum = 0;
                for (std::size_t k = 0; k < _cols; k++)
                    sum += left[k] * right[k];
                result(i, j) = sum;
            }
        }
    });
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
