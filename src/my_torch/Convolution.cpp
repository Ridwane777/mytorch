#include "my_torch/Convolution.hpp"

#include <algorithm>
#include <string>

#include "my_torch/Error.hpp"
#include "my_torch/Parallel.hpp"

namespace my_torch {

namespace {

// Samples handled together by Conv2D: the patch matrix of a block has block·outputs rows, so this
// bounds the memory whatever the batch size (the validation set is predicted 4096 rows at a time).
constexpr std::size_t blockSamples = 256;

void requireWidth(const char *type, std::size_t expected, std::size_t received)
{
    if (received != expected)
        throw Error(std::string(type) + " expects " + std::to_string(expected) +
                    " inputs but receives " + std::to_string(received));
}

void requireRows(const char *type, std::size_t expected, const Matrix &gradOutput,
                 std::size_t width)
{
    if (gradOutput.rows() != expected || gradOutput.cols() != width)
        throw Error(std::string(type) + ": backward expects a " + std::to_string(expected) + " x " +
                    std::to_string(width) + " gradient but receives " +
                    std::to_string(gradOutput.rows()) + " x " + std::to_string(gradOutput.cols()));
}

void requireShape(const char *type, const PlaneShape &shape)
{
    if (shape.channels == 0 || shape.height == 0 || shape.width == 0)
        throw Error(std::string(type) + ": channels, height and width must be positive");
}

// Size of the output along one axis for windows of `window` with a stride, after `padding` zeros on
// both sides. Throws when the window is larger than the padded input.
std::size_t outputSize(const char *type, std::size_t input, std::size_t window, std::size_t stride,
                       std::size_t padding)
{
    if (window == 0)
        throw Error(std::string(type) + ": the window size must be positive");
    if (stride == 0)
        throw Error(std::string(type) + ": the stride must be positive");
    if (window > input + 2 * padding)
        throw Error(std::string(type) + ": a window of " + std::to_string(window) +
                    " does not fit in an input of " + std::to_string(input) + " with padding " +
                    std::to_string(padding));
    return (input + 2 * padding - window) / stride + 1;
}

PlaneShape poolOutput(const char *type, const PlaneShape &input, std::size_t size,
                      std::size_t stride)
{
    requireShape(type, input);
    return {input.channels, outputSize(type, input.height, size, stride, 0),
            outputSize(type, input.width, size, stride, 0)};
}

void writeShape(std::ostream &out, const char *type, const PlaneShape &shape)
{
    out << type << " channels=" << shape.channels << " height=" << shape.height
        << " width=" << shape.width;
}

} // namespace

// ---- ToPlanes ----------------------------------------------------------------------------------

ToPlanes::ToPlanes(std::size_t inputs, std::size_t scalars, std::size_t count, std::size_t height,
                   std::size_t width)
    : _inputs(inputs), _scalars(scalars), _count(count), _height(height), _width(width), _planes(0)
{
    if (height == 0 || width == 0)
        throw Error("to_planes: height and width must be positive");
    const std::size_t cells = height * width;
    if (count == 0)
        throw Error("to_planes: count must be positive");
    if (scalars % cells != 0)
        throw Error("to_planes: scalars must be a multiple of " + std::to_string(cells) +
                    " (the planes before them)");
    if (scalars + count > inputs)
        throw Error("to_planes: the scalars do not fit in " + std::to_string(inputs) + " inputs");
    if ((inputs - count) % cells != 0 || inputs == count)
        throw Error("to_planes: inputs - count must be a positive multiple of " +
                    std::to_string(cells));
    _planes = (inputs - count) / cells;
}

Matrix ToPlanes::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("to_planes", _inputs, input.cols());
    const std::size_t cells = _height * _width;
    const std::size_t beforeScalars = _scalars / cells;
    Matrix out(input.rows(), outputShape().size());

    parallelFor(input.rows(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *in = input.data() + r * _inputs;
            Scalar *dst = out.data() + r * out.cols();
            for (std::size_t p = 0; p < _planes; p++) {
                const Scalar *plane = in + p * cells + (p >= beforeScalars ? _count : 0);
                std::copy(plane, plane + cells, dst + p * cells);
            }
            for (std::size_t j = 0; j < _count; j++)
                std::fill(dst + (_planes + j) * cells, dst + (_planes + j + 1) * cells,
                          in[_scalars + j]);
        }
    });
    return out;
}

Matrix ToPlanes::backward(const Matrix &gradOutput)
{
    requireWidth("to_planes", outputShape().size(), gradOutput.cols());
    const std::size_t cells = _height * _width;
    const std::size_t beforeScalars = _scalars / cells;
    Matrix grad(gradOutput.rows(), _inputs);

    parallelFor(gradOutput.rows(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *g = gradOutput.data() + r * gradOutput.cols();
            Scalar *dst = grad.data() + r * _inputs;
            for (std::size_t p = 0; p < _planes; p++)
                std::copy(g + p * cells, g + (p + 1) * cells,
                          dst + p * cells + (p >= beforeScalars ? _count : 0));
            for (std::size_t j = 0; j < _count; j++) {
                Scalar sum = 0;
                for (std::size_t s = 0; s < cells; s++)
                    sum += g[(_planes + j) * cells + s];
                dst[_scalars + j] = sum;
            }
        }
    });
    return grad;
}

std::size_t ToPlanes::outputWidth(std::size_t inputWidth) const
{
    requireWidth("to_planes", _inputs, inputWidth);
    return outputShape().size();
}

std::string ToPlanes::type() const
{
    return "to_planes";
}

void ToPlanes::save(std::ostream &out) const
{
    out << "to_planes inputs=" << _inputs << " scalars=" << _scalars << " count=" << _count
        << " height=" << _height << " width=" << _width << '\n';
}

std::size_t ToPlanes::inputs() const noexcept
{
    return _inputs;
}

std::size_t ToPlanes::scalars() const noexcept
{
    return _scalars;
}

std::size_t ToPlanes::count() const noexcept
{
    return _count;
}

PlaneShape ToPlanes::outputShape() const noexcept
{
    return {_planes + _count, _height, _width};
}

// ---- Conv2D ------------------------------------------------------------------------------------

namespace {

// Where the patch of output position (oy, ox) reads in a sample, shared by im2col and col2im.
struct Window {
    long top;
    long left;
};

// Rows of `cols` (outH·outW of them, one per output position in reading order) receive the patches
// of one sample, each of channels·k·k values in channel-major order; padding positions are 0.
void im2col(const Scalar *sample, const PlaneShape &in, const PlaneShape &out, std::size_t kernel,
            std::size_t stride, std::size_t padding, Scalar *cols)
{
    const std::size_t patch = in.channels * kernel * kernel;
    for (std::size_t oy = 0; oy < out.height; oy++)
        for (std::size_t ox = 0; ox < out.width; ox++) {
            Scalar *row = cols + (oy * out.width + ox) * patch;
            const Window w{static_cast<long>(oy * stride) - static_cast<long>(padding),
                           static_cast<long>(ox * stride) - static_cast<long>(padding)};
            for (std::size_t c = 0; c < in.channels; c++)
                for (std::size_t ky = 0; ky < kernel; ky++)
                    for (std::size_t kx = 0; kx < kernel; kx++) {
                        const long y = w.top + static_cast<long>(ky);
                        const long x = w.left + static_cast<long>(kx);
                        const bool inside = y >= 0 && x >= 0 && y < static_cast<long>(in.height) &&
                                            x < static_cast<long>(in.width);
                        *row++ =
                            inside
                                ? sample[(c * in.height + static_cast<std::size_t>(y)) * in.width +
                                         static_cast<std::size_t>(x)]
                                : 0;
                    }
        }
}

// The reverse of im2col: adds the gradient of each patch value to the input position it was read
// from; values that came from the padding are dropped.
void col2im(const Scalar *cols, const PlaneShape &in, const PlaneShape &out, std::size_t kernel,
            std::size_t stride, std::size_t padding, Scalar *sample)
{
    const std::size_t patch = in.channels * kernel * kernel;
    for (std::size_t oy = 0; oy < out.height; oy++)
        for (std::size_t ox = 0; ox < out.width; ox++) {
            const Scalar *row = cols + (oy * out.width + ox) * patch;
            const Window w{static_cast<long>(oy * stride) - static_cast<long>(padding),
                           static_cast<long>(ox * stride) - static_cast<long>(padding)};
            for (std::size_t c = 0; c < in.channels; c++)
                for (std::size_t ky = 0; ky < kernel; ky++)
                    for (std::size_t kx = 0; kx < kernel; kx++) {
                        const long y = w.top + static_cast<long>(ky);
                        const long x = w.left + static_cast<long>(kx);
                        const Scalar value = *row++;
                        if (y >= 0 && x >= 0 && y < static_cast<long>(in.height) &&
                            x < static_cast<long>(in.width))
                            sample[(c * in.height + static_cast<std::size_t>(y)) * in.width +
                                   static_cast<std::size_t>(x)] += value;
                    }
        }
}

} // namespace

Conv2D::Conv2D(PlaneShape input, std::size_t filters, std::size_t kernel, std::size_t stride,
               std::size_t padding, std::uint32_t seed, Init init)
    : _input(input), _kernel(kernel), _stride(stride), _padding(padding)
{
    requireShape("conv2d", input);
    if (filters == 0)
        throw Error("conv2d: filters must be positive");
    _output = {filters, outputSize("conv2d", input.height, kernel, stride, padding),
               outputSize("conv2d", input.width, kernel, stride, padding)};
    const std::size_t patch = input.channels * kernel * kernel;
    // Drawn as a Dense layer patch × filters would be (same laws, fan_in = patch), then stored one
    // filter per row.
    _weights = Dense(patch, filters, seed, {}, init).weights().transpose();
    _bias = Matrix(1, filters);
    _weightsGrad = Matrix(filters, patch);
    _biasGrad = Matrix(1, filters);
}

Matrix Conv2D::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("conv2d", _input.size(), input.cols());
    _cache = input;
    const std::size_t batch = input.rows();
    const std::size_t positions = _output.height * _output.width;
    const std::size_t filters = _output.channels;
    const std::size_t patch = _weights.cols();
    Matrix out(batch, _output.size());

    for (std::size_t first = 0; first < batch; first += blockSamples) {
        const std::size_t n = std::min(blockSamples, batch - first);
        Matrix cols(n * positions, patch);
        parallelFor(n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t s = begin; s < end; s++)
                im2col(input.data() + (first + s) * input.cols(), _input, _output, _kernel, _stride,
                       _padding, cols.data() + s * positions * patch);
        });
        const Matrix product = cols.matmulTranspose(_weights).addRowVector(_bias);
        parallelFor(n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t s = begin; s < end; s++) {
                Scalar *dst = out.data() + (first + s) * out.cols();
                for (std::size_t p = 0; p < positions; p++)
                    for (std::size_t f = 0; f < filters; f++)
                        dst[f * positions + p] = product(s * positions + p, f);
            }
        });
    }
    return out;
}

Matrix Conv2D::backward(const Matrix &gradOutput)
{
    const std::size_t batch = _cache.rows();
    requireRows("conv2d", batch, gradOutput, _output.size());
    const std::size_t positions = _output.height * _output.width;
    const std::size_t filters = _output.channels;
    const std::size_t patch = _weights.cols();
    Matrix gradInput(batch, _input.size());

    for (std::size_t first = 0; first < batch; first += blockSamples) {
        const std::size_t n = std::min(blockSamples, batch - first);
        // dZ in patch order: one row per (sample, position), one column per filter.
        Matrix grad(n * positions, filters);
        Matrix cols(n * positions, patch);
        parallelFor(n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t s = begin; s < end; s++) {
                const Scalar *g = gradOutput.data() + (first + s) * gradOutput.cols();
                for (std::size_t p = 0; p < positions; p++)
                    for (std::size_t f = 0; f < filters; f++)
                        grad(s * positions + p, f) = g[f * positions + p];
                im2col(_cache.data() + (first + s) * _cache.cols(), _input, _output, _kernel,
                       _stride, _padding, cols.data() + s * positions * patch);
            }
        });
        _weightsGrad += grad.transposeMatmul(cols);
        _biasGrad += grad.sumRows();
        const Matrix gradCols = grad.matmul(_weights);
        parallelFor(n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t s = begin; s < end; s++)
                col2im(gradCols.data() + s * positions * patch, _input, _output, _kernel, _stride,
                       _padding, gradInput.data() + (first + s) * gradInput.cols());
        });
    }
    return gradInput;
}

std::vector<Parameter> Conv2D::parameters()
{
    return {{&_weights, &_weightsGrad}, {&_bias, &_biasGrad}};
}

std::size_t Conv2D::outputWidth(std::size_t inputWidth) const
{
    requireWidth("conv2d", _input.size(), inputWidth);
    return _output.size();
}

std::string Conv2D::type() const
{
    return "conv2d";
}

void Conv2D::save(std::ostream &out) const
{
    writeShape(out, "conv2d", _input);
    out << " filters=" << _output.channels << " kernel=" << _kernel << " stride=" << _stride
        << " padding=" << _padding << '\n';
    for (std::size_t r = 0; r < _weights.rows(); r++)
        for (std::size_t c = 0; c < _weights.cols(); c++)
            out << _weights(r, c) << (c + 1 < _weights.cols() ? ' ' : '\n');
    for (std::size_t c = 0; c < _bias.cols(); c++)
        out << _bias(0, c) << (c + 1 < _bias.cols() ? ' ' : '\n');
}

Matrix &Conv2D::weights() noexcept
{
    return _weights;
}

const Matrix &Conv2D::weights() const noexcept
{
    return _weights;
}

Matrix &Conv2D::bias() noexcept
{
    return _bias;
}

PlaneShape Conv2D::inputShape() const noexcept
{
    return _input;
}

PlaneShape Conv2D::outputShape() const noexcept
{
    return _output;
}

std::size_t Conv2D::filters() const noexcept
{
    return _output.channels;
}

std::size_t Conv2D::kernel() const noexcept
{
    return _kernel;
}

std::size_t Conv2D::stride() const noexcept
{
    return _stride;
}

std::size_t Conv2D::padding() const noexcept
{
    return _padding;
}

// ---- MaxPool2D ---------------------------------------------------------------------------------

MaxPool2D::MaxPool2D(PlaneShape input, std::size_t size, std::size_t stride)
    : _input(input), _output(poolOutput("max_pool2d", input, size, stride)), _size(size),
      _stride(stride)
{
}

Matrix MaxPool2D::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("max_pool2d", _input.size(), input.cols());
    _rows = input.rows();
    Matrix out(_rows, _output.size());
    _argmax.assign(_rows * _output.size(), 0);

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *in = input.data() + r * input.cols();
            std::size_t cell = r * _output.size();
            for (std::size_t c = 0; c < _input.channels; c++)
                for (std::size_t oy = 0; oy < _output.height; oy++)
                    for (std::size_t ox = 0; ox < _output.width; ox++, cell++) {
                        std::size_t best =
                            (c * _input.height + oy * _stride) * _input.width + ox * _stride;
                        for (std::size_t ky = 0; ky < _size; ky++)
                            for (std::size_t kx = 0; kx < _size; kx++) {
                                const std::size_t at =
                                    (c * _input.height + oy * _stride + ky) * _input.width +
                                    ox * _stride + kx;
                                if (in[at] > in[best]) // strict: the first maximum wins
                                    best = at;
                            }
                        _argmax[cell] = best;
                        out.data()[cell] = in[best];
                    }
        }
    });
    return out;
}

Matrix MaxPool2D::backward(const Matrix &gradOutput)
{
    requireRows("max_pool2d", _rows, gradOutput, _output.size());
    Matrix gradInput(_rows, _input.size());

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++)
            for (std::size_t cell = r * _output.size(); cell < (r + 1) * _output.size(); cell++)
                gradInput.data()[r * _input.size() + _argmax[cell]] += gradOutput.data()[cell];
    });
    return gradInput;
}

std::size_t MaxPool2D::outputWidth(std::size_t inputWidth) const
{
    requireWidth("max_pool2d", _input.size(), inputWidth);
    return _output.size();
}

std::string MaxPool2D::type() const
{
    return "max_pool2d";
}

void MaxPool2D::save(std::ostream &out) const
{
    writeShape(out, "max_pool2d", _input);
    out << " size=" << _size << " stride=" << _stride << '\n';
}

PlaneShape MaxPool2D::inputShape() const noexcept
{
    return _input;
}

PlaneShape MaxPool2D::outputShape() const noexcept
{
    return _output;
}

std::size_t MaxPool2D::size() const noexcept
{
    return _size;
}

std::size_t MaxPool2D::stride() const noexcept
{
    return _stride;
}

// ---- AvgPool2D ---------------------------------------------------------------------------------

AvgPool2D::AvgPool2D(PlaneShape input, std::size_t size, std::size_t stride)
    : _input(input), _output(poolOutput("avg_pool2d", input, size, stride)), _size(size),
      _stride(stride)
{
}

Matrix AvgPool2D::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("avg_pool2d", _input.size(), input.cols());
    _rows = input.rows();
    Matrix out(_rows, _output.size());
    const Scalar share = 1.0 / static_cast<Scalar>(_size * _size);

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *in = input.data() + r * input.cols();
            std::size_t cell = r * _output.size();
            for (std::size_t c = 0; c < _input.channels; c++)
                for (std::size_t oy = 0; oy < _output.height; oy++)
                    for (std::size_t ox = 0; ox < _output.width; ox++, cell++) {
                        Scalar sum = 0;
                        for (std::size_t ky = 0; ky < _size; ky++)
                            for (std::size_t kx = 0; kx < _size; kx++)
                                sum += in[(c * _input.height + oy * _stride + ky) * _input.width +
                                          ox * _stride + kx];
                        out.data()[cell] = sum * share;
                    }
        }
    });
    return out;
}

Matrix AvgPool2D::backward(const Matrix &gradOutput)
{
    requireRows("avg_pool2d", _rows, gradOutput, _output.size());
    Matrix gradInput(_rows, _input.size());
    const Scalar share = 1.0 / static_cast<Scalar>(_size * _size);

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            Scalar *dst = gradInput.data() + r * _input.size();
            std::size_t cell = r * _output.size();
            for (std::size_t c = 0; c < _input.channels; c++)
                for (std::size_t oy = 0; oy < _output.height; oy++)
                    for (std::size_t ox = 0; ox < _output.width; ox++, cell++)
                        for (std::size_t ky = 0; ky < _size; ky++)
                            for (std::size_t kx = 0; kx < _size; kx++)
                                dst[(c * _input.height + oy * _stride + ky) * _input.width +
                                    ox * _stride + kx] += gradOutput.data()[cell] * share;
        }
    });
    return gradInput;
}

std::size_t AvgPool2D::outputWidth(std::size_t inputWidth) const
{
    requireWidth("avg_pool2d", _input.size(), inputWidth);
    return _output.size();
}

std::string AvgPool2D::type() const
{
    return "avg_pool2d";
}

void AvgPool2D::save(std::ostream &out) const
{
    writeShape(out, "avg_pool2d", _input);
    out << " size=" << _size << " stride=" << _stride << '\n';
}

PlaneShape AvgPool2D::inputShape() const noexcept
{
    return _input;
}

PlaneShape AvgPool2D::outputShape() const noexcept
{
    return _output;
}

std::size_t AvgPool2D::size() const noexcept
{
    return _size;
}

std::size_t AvgPool2D::stride() const noexcept
{
    return _stride;
}

// ---- GlobalAvgPool -----------------------------------------------------------------------------

GlobalAvgPool::GlobalAvgPool(PlaneShape input) : _input(input)
{
    requireShape("global_avg_pool", input);
}

Matrix GlobalAvgPool::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("global_avg_pool", _input.size(), input.cols());
    _rows = input.rows();
    const std::size_t plane = _input.height * _input.width;
    Matrix out(_rows, _input.channels);

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++)
            for (std::size_t c = 0; c < _input.channels; c++) {
                Scalar sum = 0;
                const Scalar *p = input.data() + r * input.cols() + c * plane;
                for (std::size_t i = 0; i < plane; i++)
                    sum += p[i];
                out(r, c) = sum / static_cast<Scalar>(plane);
            }
    });
    return out;
}

Matrix GlobalAvgPool::backward(const Matrix &gradOutput)
{
    requireRows("global_avg_pool", _rows, gradOutput, _input.channels);
    const std::size_t plane = _input.height * _input.width;
    Matrix gradInput(_rows, _input.size());

    parallelFor(_rows, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++)
            for (std::size_t c = 0; c < _input.channels; c++)
                std::fill(gradInput.data() + r * _input.size() + c * plane,
                          gradInput.data() + r * _input.size() + (c + 1) * plane,
                          gradOutput(r, c) / static_cast<Scalar>(plane));
    });
    return gradInput;
}

std::size_t GlobalAvgPool::outputWidth(std::size_t inputWidth) const
{
    requireWidth("global_avg_pool", _input.size(), inputWidth);
    return _input.channels;
}

std::string GlobalAvgPool::type() const
{
    return "global_avg_pool";
}

void GlobalAvgPool::save(std::ostream &out) const
{
    writeShape(out, "global_avg_pool", _input);
    out << '\n';
}

PlaneShape GlobalAvgPool::inputShape() const noexcept
{
    return _input;
}

PlaneShape GlobalAvgPool::outputShape() const noexcept
{
    return {_input.channels, 1, 1};
}

} // namespace my_torch
