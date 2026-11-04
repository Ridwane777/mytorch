#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "my_torch/Dense.hpp"
#include "my_torch/Layer.hpp"

namespace my_torch {

// Convolutional layers (ADR-0037). A Matrix row is one sample, and an image-like sample is stored
// flattened channel-major, channels × height × width: the value of channel c at row y, column x
// sits in column (c·height + y)·width + x. The layers do not infer the geometry from the width of
// their input: it is given to the constructor (the configuration and network file write it), and a
// layer throws my_torch::Error when the input it receives has another width.

// Dimensions of a stack of planes.
struct PlaneShape {
    std::size_t channels = 0;
    std::size_t height = 0;
    std::size_t width = 0;

    std::size_t size() const noexcept
    {
        return channels * height * width;
    }
};

// Parameter-free reshape of a flat row into image planes of height × width (ADR-0037, made generic
// by ADR-0041). The input row holds `inputs` columns: planes of height·width columns (index
// plane·height·width + cell), except for `count` scalars that start at column `scalars`, on a plane
// boundary. The output is channels × height × width with the planes in their order, then one more
// plane per scalar, the scalar repeated on every cell. In backward, a cell receives its own gradient
// and a scalar the sum over its plane. The layer knows nothing of what the planes hold: the
// configuration gives the layout of its encoding.
class ToPlanes : public Layer {
  public:
    // Throws my_torch::Error unless height and width are positive, count > 0, scalars is a
    // multiple of height·width, scalars + count <= inputs, and inputs - count is a positive
    // multiple of height·width.
    ToPlanes(std::size_t inputs, std::size_t scalars, std::size_t count, std::size_t height,
             std::size_t width);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "to_planes inputs=… scalars=… count=… height=… width=…"
    void save(std::ostream &out) const override;

    std::size_t inputs() const noexcept;
    std::size_t scalars() const noexcept;
    std::size_t count() const noexcept;
    PlaneShape outputShape() const noexcept;

  private:
    std::size_t _inputs;
    std::size_t _scalars;
    std::size_t _count;
    std::size_t _height;
    std::size_t _width;
    std::size_t _planes;
};

// 2D convolution (cross-correlation, as in every library): `filters` kernels of channels × k × k
// slide over the input with the given stride and zero padding on the four sides, and each adds one
// bias. The output has `filters` channels of ((height + 2·padding - k) / stride + 1) rows and the
// same formula for the columns (the division rounds down).
//
// The weights are a matrix filters × (channels·k·k), one row per filter in channel-major order
// (the network file writes them this way), the bias a row of `filters`. Forward builds the patches
// of the input as rows (im2col), computes one product with the transposed weights, and puts the
// result back in channel-major order; backward does the product for dW and dPatches, and adds the
// patches' gradients back to the input (col2im). Samples are handled by blocks to bound the memory.
class Conv2D : public Layer {
  public:
    // Weights drawn from std::mt19937(seed) following `init` with fan_in = channels·k·k and
    // fan_out = filters (He normal by default), biases at zero. Throws my_torch::Error on a zero
    // size, a zero stride, or a kernel larger than the padded input.
    Conv2D(PlaneShape input, std::size_t filters, std::size_t kernel, std::size_t stride,
           std::size_t padding, std::uint32_t seed = 42, Init init = Init::HeNormal);

    Matrix forward(const Matrix &input, bool training) override;
    // dW += dZᵀ·patches, db += per-filter sum of dZ, returns dX. Gradients accumulate, as Dense.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "conv2d channels=… height=… width=… filters=… kernel=… stride=… padding=…", then one line
    // of weights per filter, then the biases.
    void save(std::ostream &out) const override;

    Matrix &weights() noexcept;
    const Matrix &weights() const noexcept;
    Matrix &bias() noexcept;
    PlaneShape inputShape() const noexcept;
    PlaneShape outputShape() const noexcept;
    std::size_t filters() const noexcept;
    std::size_t kernel() const noexcept;
    std::size_t stride() const noexcept;
    std::size_t padding() const noexcept;

  private:
    PlaneShape _input;
    PlaneShape _output;
    std::size_t _kernel;
    std::size_t _stride;
    std::size_t _padding;
    Matrix _weights;
    Matrix _bias;
    Matrix _weightsGrad;
    Matrix _biasGrad;
    Matrix _cache; // input of the last forward
};

// Max pooling over square windows of `size` with the given stride, no padding, per channel. The
// output has ((height - size) / stride + 1) rows, rounded down (a last incomplete window is
// ignored), and the same for the columns. Backward sends each gradient to the position of the
// maximum, the first one in reading order when several are equal.
class MaxPool2D : public Layer {
  public:
    // Throws my_torch::Error on a zero size or stride, or a window larger than the input.
    MaxPool2D(PlaneShape input, std::size_t size, std::size_t stride);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "max_pool2d channels=… height=… width=… size=… stride=…"
    void save(std::ostream &out) const override;

    PlaneShape inputShape() const noexcept;
    PlaneShape outputShape() const noexcept;
    std::size_t size() const noexcept;
    std::size_t stride() const noexcept;

  private:
    PlaneShape _input;
    PlaneShape _output;
    std::size_t _size;
    std::size_t _stride;
    std::size_t _rows = 0;
    std::vector<std::size_t> _argmax; // per output cell of the last forward, column of the input
};

// Average pooling, same windows as MaxPool2D; backward shares each gradient equally between the
// size² positions of its window.
class AvgPool2D : public Layer {
  public:
    // Throws my_torch::Error on a zero size or stride, or a window larger than the input.
    AvgPool2D(PlaneShape input, std::size_t size, std::size_t stride);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "avg_pool2d channels=… height=… width=… size=… stride=…"
    void save(std::ostream &out) const override;

    PlaneShape inputShape() const noexcept;
    PlaneShape outputShape() const noexcept;
    std::size_t size() const noexcept;
    std::size_t stride() const noexcept;

  private:
    PlaneShape _input;
    PlaneShape _output;
    std::size_t _size;
    std::size_t _stride;
    std::size_t _rows = 0;
};

// Mean of each channel over its whole plane: channels × height × width in, `channels` out.
class GlobalAvgPool : public Layer {
  public:
    // Throws my_torch::Error on a zero dimension.
    explicit GlobalAvgPool(PlaneShape input);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "global_avg_pool channels=… height=… width=…"
    void save(std::ostream &out) const override;

    PlaneShape inputShape() const noexcept;
    PlaneShape outputShape() const noexcept;

  private:
    PlaneShape _input;
    std::size_t _rows = 0;
};

} // namespace my_torch
