#include "my_torch/Dropout.hpp"

#include "my_torch/Error.hpp"

namespace my_torch {

Dropout::Dropout(Scalar rate, std::uint64_t seed) : _rate(rate), _rng(seed)
{
    if (!(rate >= 0 && rate < 1))
        throw Error("dropout: rate must be in [0, 1[");
}

Matrix Dropout::forward(const Matrix &input, bool training)
{
    if (!training || _rate == 0) {
        _mask = Matrix();
        return input;
    }
    const Scalar keep = 1 / (1 - _rate);
    _mask = Matrix(input.rows(), input.cols());
    for (std::size_t i = 0; i < _mask.size(); i++) {
        // Uniform in [0, 1) from the 53 high bits: fully specified, unlike
        // std::uniform_real_distribution, so the mask is the same on every toolchain.
        const Scalar draw = static_cast<Scalar>(_rng() >> 11) * 0x1.0p-53;
        _mask.data()[i] = draw < _rate ? 0 : keep;
    }
    return input.hadamard(_mask);
}

Matrix Dropout::backward(const Matrix &gradOutput)
{
    if (_mask.size() == 0)
        return gradOutput;
    return gradOutput.hadamard(_mask);
}

std::string Dropout::type() const
{
    return "dropout";
}

Scalar Dropout::rate() const noexcept
{
    return _rate;
}

} // namespace my_torch
