// Token layers and self-attention (ADR-0047).
#include "my_torch/Attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "my_torch/Error.hpp"
#include "my_torch/Parallel.hpp"

namespace my_torch {

namespace {

// Standard deviation of the position table at the start of training (ADR-0047).
constexpr Scalar positionalSpread = 0.1;

void requireWidth(const char *type, std::size_t expected, std::size_t received)
{
    if (received != expected)
        throw Error(std::string(type) + " expects " + std::to_string(expected) +
                    " inputs but receives " + std::to_string(received));
}

void requireGradient(const char *type, std::size_t rows, std::size_t width, const Matrix &gradient)
{
    if (gradient.rows() != rows || gradient.cols() != width)
        throw Error(std::string(type) + ": backward expects a " + std::to_string(rows) + " x " +
                    std::to_string(width) + " gradient but receives " +
                    std::to_string(gradient.rows()) + " x " + std::to_string(gradient.cols()));
}

void requirePositive(const char *type, const char *name, std::size_t value)
{
    if (value == 0)
        throw Error(std::string(type) + ": " + name + " must be positive");
}

// The same numbers with another shape (row-major storage): a batch × (tokens·width) matrix as the
// (batch·tokens) × width matrix of its tokens, and back.
Matrix reshaped(const Matrix &source, std::size_t rows, std::size_t cols)
{
    Matrix out(rows, cols);
    std::copy(source.data(), source.data() + source.size(), out.data());
    return out;
}

void writeRows(std::ostream &out, const Matrix &matrix)
{
    for (std::size_t r = 0; r < matrix.rows(); r++)
        for (std::size_t c = 0; c < matrix.cols(); c++)
            out << matrix(r, c) << (c + 1 < matrix.cols() ? ' ' : '\n');
}

} // namespace

// ---- ToTokens ----------------------------------------------------------------------------------

ToTokens::ToTokens(std::size_t inputs, std::size_t scalars, std::size_t count, std::size_t tokens)
    : _inputs(inputs), _scalars(scalars), _count(count), _tokens(tokens), _planes(0)
{
    requirePositive("to_tokens", "tokens", tokens);
    requirePositive("to_tokens", "count", count);
    if (scalars % tokens != 0)
        throw Error("to_tokens: scalars must be a multiple of " + std::to_string(tokens) +
                    " (the planes before them)");
    if (scalars + count > inputs)
        throw Error("to_tokens: the scalars do not fit in " + std::to_string(inputs) + " inputs");
    if ((inputs - count) % tokens != 0 || inputs == count)
        throw Error("to_tokens: inputs - count must be a positive multiple of " +
                    std::to_string(tokens));
    _planes = (inputs - count) / tokens;
}

Matrix ToTokens::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("to_tokens", _inputs, input.cols());
    const std::size_t beforeScalars = _scalars / _tokens;
    const std::size_t width = features();
    Matrix out(input.rows(), _tokens * width);

    parallelFor(input.rows(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *in = input.data() + r * _inputs;
            Scalar *dst = out.data() + r * out.cols();
            for (std::size_t t = 0; t < _tokens; t++, dst += width) {
                for (std::size_t p = 0; p < _planes; p++)
                    dst[p] = in[p * _tokens + t + (p >= beforeScalars ? _count : 0)];
                for (std::size_t j = 0; j < _count; j++)
                    dst[_planes + j] = in[_scalars + j];
            }
        }
    });
    return out;
}

Matrix ToTokens::backward(const Matrix &gradOutput)
{
    const std::size_t width = features();
    requireWidth("to_tokens", _tokens * width, gradOutput.cols());
    const std::size_t beforeScalars = _scalars / _tokens;
    Matrix grad(gradOutput.rows(), _inputs);

    parallelFor(gradOutput.rows(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const Scalar *g = gradOutput.data() + r * gradOutput.cols();
            Scalar *dst = grad.data() + r * _inputs;
            for (std::size_t t = 0; t < _tokens; t++, g += width) {
                for (std::size_t p = 0; p < _planes; p++)
                    dst[p * _tokens + t + (p >= beforeScalars ? _count : 0)] = g[p];
                for (std::size_t j = 0; j < _count; j++)
                    dst[_scalars + j] += g[_planes + j];
            }
        }
    });
    return grad;
}

std::size_t ToTokens::outputWidth(std::size_t inputWidth) const
{
    requireWidth("to_tokens", _inputs, inputWidth);
    return _tokens * features();
}

std::string ToTokens::type() const
{
    return "to_tokens";
}

void ToTokens::save(std::ostream &out) const
{
    out << "to_tokens inputs=" << _inputs << " scalars=" << _scalars << " count=" << _count
        << " tokens=" << _tokens << '\n';
}

std::size_t ToTokens::inputs() const noexcept
{
    return _inputs;
}

std::size_t ToTokens::scalars() const noexcept
{
    return _scalars;
}

std::size_t ToTokens::count() const noexcept
{
    return _count;
}

std::size_t ToTokens::tokens() const noexcept
{
    return _tokens;
}

std::size_t ToTokens::features() const noexcept
{
    return _planes + _count;
}

// ---- TokenDense --------------------------------------------------------------------------------

TokenDense::TokenDense(std::size_t tokens, std::size_t inputs, std::size_t outputs,
                       std::uint32_t seed, Regularization regularization, Init init)
    : _tokens(tokens), _regularization(regularization)
{
    requirePositive("token_dense", "tokens", tokens);
    requirePositive("token_dense", "inputs", inputs);
    requirePositive("token_dense", "outputs", outputs);
    // Drawn as a Dense layer inputs × outputs would be (same laws, fan_in = inputs).
    _weights = Dense(inputs, outputs, seed, regularization, init).weights();
    _bias = Matrix(1, outputs);
    _weightsGrad = Matrix(inputs, outputs);
    _biasGrad = Matrix(1, outputs);
}

Matrix TokenDense::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("token_dense", _tokens * _weights.rows(), input.cols());
    _input = reshaped(input, input.rows() * _tokens, _weights.rows());
    const Matrix out = _input.matmul(_weights).addRowVector(_bias);
    return reshaped(out, input.rows(), _tokens * _weights.cols());
}

Matrix TokenDense::backward(const Matrix &gradOutput)
{
    requireGradient("token_dense", _input.rows() / _tokens, _tokens * _weights.cols(), gradOutput);
    const Matrix g = reshaped(gradOutput, gradOutput.rows() * _tokens, _weights.cols());
    _weightsGrad += _input.transposeMatmul(g);
    _biasGrad += g.sumRows();
    if (_regularization.l1 > 0 || _regularization.l2 > 0)
        for (std::size_t i = 0; i < _weights.size(); i++) {
            const Scalar w = _weights.data()[i];
            const Scalar sign = w > 0 ? 1.0 : (w < 0 ? -1.0 : 0.0);
            _weightsGrad.data()[i] += _regularization.l1 * sign + 2 * _regularization.l2 * w;
        }
    return reshaped(g.matmulTranspose(_weights), gradOutput.rows(), _tokens * _weights.rows());
}

std::vector<Parameter> TokenDense::parameters()
{
    return {{&_weights, &_weightsGrad}, {&_bias, &_biasGrad}};
}

std::size_t TokenDense::outputWidth(std::size_t inputWidth) const
{
    requireWidth("token_dense", _tokens * _weights.rows(), inputWidth);
    return _tokens * _weights.cols();
}

std::string TokenDense::type() const
{
    return "token_dense";
}

Scalar TokenDense::penalty() const
{
    Scalar total = 0;

    for (std::size_t i = 0; i < _weights.size(); i++) {
        const Scalar w = _weights.data()[i];
        total += _regularization.l1 * std::abs(w) + _regularization.l2 * w * w;
    }
    return total;
}

void TokenDense::save(std::ostream &out) const
{
    out << "token_dense tokens=" << _tokens << " inputs=" << _weights.rows()
        << " outputs=" << _weights.cols() << " l1=" << _regularization.l1
        << " l2=" << _regularization.l2 << '\n';
    writeRows(out, _weights);
    writeRows(out, _bias);
}

std::size_t TokenDense::tokens() const noexcept
{
    return _tokens;
}

Matrix &TokenDense::weights() noexcept
{
    return _weights;
}

const Matrix &TokenDense::weights() const noexcept
{
    return _weights;
}

Matrix &TokenDense::bias() noexcept
{
    return _bias;
}

const Regularization &TokenDense::regularization() const noexcept
{
    return _regularization;
}

// ---- Positional --------------------------------------------------------------------------------

Positional::Positional(std::size_t tokens, std::size_t dim, std::uint32_t seed)
    : _table(tokens, dim), _tableGrad(tokens, dim)
{
    requirePositive("positional", "tokens", tokens);
    requirePositive("positional", "dim", dim);
    std::mt19937 rng(seed);
    std::normal_distribution<Scalar> dist(0, positionalSpread);
    for (std::size_t i = 0; i < _table.size(); i++)
        _table.data()[i] = dist(rng);
}

Matrix Positional::forward(const Matrix &input, bool /*training*/)
{
    requireWidth("positional", _table.size(), input.cols());
    Matrix out = input;
    const std::size_t width = _table.size();
    for (std::size_t r = 0; r < out.rows(); r++) {
        Scalar *row = out.data() + r * width;
        for (std::size_t j = 0; j < width; j++)
            row[j] += _table.data()[j];
    }
    return out;
}

Matrix Positional::backward(const Matrix &gradOutput)
{
    requireWidth("positional", _table.size(), gradOutput.cols());
    const std::size_t width = _table.size();
    for (std::size_t r = 0; r < gradOutput.rows(); r++) {
        const Scalar *row = gradOutput.data() + r * width;
        for (std::size_t j = 0; j < width; j++)
            _tableGrad.data()[j] += row[j];
    }
    return gradOutput;
}

std::vector<Parameter> Positional::parameters()
{
    return {{&_table, &_tableGrad}};
}

std::size_t Positional::outputWidth(std::size_t inputWidth) const
{
    requireWidth("positional", _table.size(), inputWidth);
    return inputWidth;
}

std::string Positional::type() const
{
    return "positional";
}

void Positional::save(std::ostream &out) const
{
    out << "positional tokens=" << _table.rows() << " dim=" << _table.cols() << '\n';
    writeRows(out, _table);
}

std::size_t Positional::tokens() const noexcept
{
    return _table.rows();
}

std::size_t Positional::dim() const noexcept
{
    return _table.cols();
}

Matrix &Positional::table() noexcept
{
    return _table;
}

const Matrix &Positional::table() const noexcept
{
    return _table;
}

// ---- SelfAttention -----------------------------------------------------------------------------

namespace {

// Attention matrix P of head `head` of sample `sample`, written in p (tokens × tokens): the scores
// Q·Kᵀ / sqrt(headDim) of the head's columns, then a softmax per row (the maximum subtracted).
void attentionMatrix(const Matrix &q, const Matrix &k, std::size_t sample, std::size_t head,
                     std::size_t tokens, std::size_t dim, std::size_t headDim, Scalar *p)
{
    const Scalar scale = 1 / std::sqrt(static_cast<Scalar>(headDim));
    const std::size_t offset = head * headDim;

    for (std::size_t t = 0; t < tokens; t++) {
        const Scalar *query = q.data() + (sample * tokens + t) * dim + offset;
        Scalar *row = p + t * tokens;
        Scalar largest = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t u = 0; u < tokens; u++) {
            const Scalar *key = k.data() + (sample * tokens + u) * dim + offset;
            Scalar score = 0;
            for (std::size_t d = 0; d < headDim; d++)
                score += query[d] * key[d];
            row[u] = score * scale;
            largest = std::max(largest, row[u]);
        }
        Scalar sum = 0;
        for (std::size_t u = 0; u < tokens; u++) {
            row[u] = std::exp(row[u] - largest);
            sum += row[u];
        }
        for (std::size_t u = 0; u < tokens; u++)
            row[u] /= sum;
    }
}

} // namespace

SelfAttention::SelfAttention(std::size_t tokens, std::size_t dim, std::size_t heads,
                             std::uint32_t seed)
    : _tokens(tokens), _dim(dim), _heads(heads)
{
    requirePositive("self_attention", "tokens", tokens);
    requirePositive("self_attention", "dim", dim);
    requirePositive("self_attention", "heads", heads);
    if (dim % heads != 0)
        throw Error("self_attention: dim (" + std::to_string(dim) +
                    ") must be a multiple of heads (" + std::to_string(heads) + ")");
    for (std::uint32_t i = 0; i < 4; i++) {
        _weights[i] = Dense(dim, dim, seed + i, {}, Init::XavierUniform).weights();
        _bias[i] = Matrix(1, dim);
        _weightsGrad[i] = Matrix(dim, dim);
        _biasGrad[i] = Matrix(1, dim);
    }
}

Matrix SelfAttention::forward(const Matrix &input, bool training)
{
    requireWidth("self_attention", _tokens * _dim, input.cols());
    const std::size_t batch = input.rows();
    const std::size_t headDim = _dim / _heads;
    _input = reshaped(input, batch * _tokens, _dim);
    _q = _input.matmul(_weights[0]).addRowVector(_bias[0]);
    _k = _input.matmul(_weights[1]).addRowVector(_bias[1]);
    _v = _input.matmul(_weights[2]).addRowVector(_bias[2]);
    _context = Matrix(batch * _tokens, _dim);
    _attention = training ? Matrix(batch * _heads * _tokens, _tokens) : Matrix();

    parallelFor(batch, [&](std::size_t begin, std::size_t end) {
        std::vector<Scalar> scratch(training ? 0 : _tokens * _tokens);
        for (std::size_t b = begin; b < end; b++)
            for (std::size_t h = 0; h < _heads; h++) {
                Scalar *p = training ? _attention.data() + (b * _heads + h) * _tokens * _tokens
                                     : scratch.data();
                attentionMatrix(_q, _k, b, h, _tokens, _dim, headDim, p);
                for (std::size_t t = 0; t < _tokens; t++) {
                    Scalar *out = _context.data() + (b * _tokens + t) * _dim + h * headDim;
                    for (std::size_t u = 0; u < _tokens; u++) {
                        const Scalar weight = p[t * _tokens + u];
                        const Scalar *value = _v.data() + (b * _tokens + u) * _dim + h * headDim;
                        for (std::size_t d = 0; d < headDim; d++)
                            out[d] += weight * value[d];
                    }
                }
            }
    });
    return reshaped(_context.matmul(_weights[3]).addRowVector(_bias[3]), batch, _tokens * _dim);
}

Matrix SelfAttention::backward(const Matrix &gradOutput)
{
    requireGradient("self_attention", _input.rows() / _tokens, _tokens * _dim, gradOutput);
    const std::size_t batch = gradOutput.rows();
    const std::size_t headDim = _dim / _heads;
    const Scalar scale = 1 / std::sqrt(static_cast<Scalar>(headDim));
    const Matrix g = reshaped(gradOutput, batch * _tokens, _dim);

    // Output projection: dWo, dbo and the gradient of the context.
    _weightsGrad[3] += _context.transposeMatmul(g);
    _biasGrad[3] += g.sumRows();
    const Matrix dContext = g.matmulTranspose(_weights[3]);

    Matrix dq(batch * _tokens, _dim);
    Matrix dk(batch * _tokens, _dim);
    Matrix dv(batch * _tokens, _dim);
    const bool kept = _attention.size() > 0;

    parallelFor(batch, [&](std::size_t begin, std::size_t end) {
        std::vector<Scalar> recomputed(kept ? 0 : _tokens * _tokens);
        std::vector<Scalar> dScores(_tokens * _tokens);
        for (std::size_t b = begin; b < end; b++)
            for (std::size_t h = 0; h < _heads; h++) {
                const std::size_t offset = h * headDim;
                const Scalar *p = nullptr;
                if (kept) {
                    p = _attention.data() + (b * _heads + h) * _tokens * _tokens;
                } else {
                    attentionMatrix(_q, _k, b, h, _tokens, _dim, headDim, recomputed.data());
                    p = recomputed.data();
                }
                const auto at = [&](const Matrix &m, std::size_t token) {
                    return m.data() + (b * _tokens + token) * _dim + offset;
                };
                // dP = dC·Vᵀ (kept in dScores), dV = Pᵀ·dC.
                for (std::size_t t = 0; t < _tokens; t++) {
                    const Scalar *dc = at(dContext, t);
                    for (std::size_t u = 0; u < _tokens; u++) {
                        const Scalar *value = at(_v, u);
                        Scalar sum = 0;
                        for (std::size_t d = 0; d < headDim; d++)
                            sum += dc[d] * value[d];
                        dScores[t * _tokens + u] = sum;
                    }
                }
                for (std::size_t u = 0; u < _tokens; u++) {
                    Scalar *dvalue = dv.data() + (b * _tokens + u) * _dim + offset;
                    for (std::size_t t = 0; t < _tokens; t++) {
                        const Scalar weight = p[t * _tokens + u];
                        const Scalar *dc = at(dContext, t);
                        for (std::size_t d = 0; d < headDim; d++)
                            dvalue[d] += weight * dc[d];
                    }
                }
                // dS = P ∘ (dP - rowsum(dP ∘ P)), already divided by sqrt(headDim).
                for (std::size_t t = 0; t < _tokens; t++) {
                    Scalar dot = 0;
                    for (std::size_t u = 0; u < _tokens; u++)
                        dot += dScores[t * _tokens + u] * p[t * _tokens + u];
                    for (std::size_t u = 0; u < _tokens; u++)
                        dScores[t * _tokens + u] =
                            p[t * _tokens + u] * (dScores[t * _tokens + u] - dot) * scale;
                }
                // dQ = dS·K, dK = dSᵀ·Q.
                for (std::size_t t = 0; t < _tokens; t++) {
                    Scalar *dquery = dq.data() + (b * _tokens + t) * _dim + offset;
                    for (std::size_t u = 0; u < _tokens; u++) {
                        const Scalar weight = dScores[t * _tokens + u];
                        const Scalar *key = at(_k, u);
                        for (std::size_t d = 0; d < headDim; d++)
                            dquery[d] += weight * key[d];
                    }
                }
                for (std::size_t u = 0; u < _tokens; u++) {
                    Scalar *dkey = dk.data() + (b * _tokens + u) * _dim + offset;
                    for (std::size_t t = 0; t < _tokens; t++) {
                        const Scalar weight = dScores[t * _tokens + u];
                        const Scalar *query = at(_q, t);
                        for (std::size_t d = 0; d < headDim; d++)
                            dkey[d] += weight * query[d];
                    }
                }
            }
    });

    // Input projections: dW = Xᵀ·dZ, db = column sums, and dX is the sum of the three paths.
    const Matrix *gradients[3] = {&dq, &dk, &dv};
    Matrix dx(batch * _tokens, _dim);
    for (std::size_t i = 0; i < 3; i++) {
        _weightsGrad[i] += _input.transposeMatmul(*gradients[i]);
        _biasGrad[i] += gradients[i]->sumRows();
        dx += gradients[i]->matmulTranspose(_weights[i]);
    }
    return reshaped(dx, batch, _tokens * _dim);
}

std::vector<Parameter> SelfAttention::parameters()
{
    std::vector<Parameter> out;
    for (std::size_t i = 0; i < 4; i++) {
        out.push_back({&_weights[i], &_weightsGrad[i]});
        out.push_back({&_bias[i], &_biasGrad[i]});
    }
    return out;
}

std::size_t SelfAttention::outputWidth(std::size_t inputWidth) const
{
    requireWidth("self_attention", _tokens * _dim, inputWidth);
    return inputWidth;
}

std::string SelfAttention::type() const
{
    return "self_attention";
}

void SelfAttention::save(std::ostream &out) const
{
    out << "self_attention tokens=" << _tokens << " dim=" << _dim << " heads=" << _heads << '\n';
    for (std::size_t i = 0; i < 4; i++) {
        writeRows(out, _weights[i]);
        writeRows(out, _bias[i]);
    }
}

std::size_t SelfAttention::tokens() const noexcept
{
    return _tokens;
}

std::size_t SelfAttention::dim() const noexcept
{
    return _dim;
}

std::size_t SelfAttention::heads() const noexcept
{
    return _heads;
}

Matrix &SelfAttention::weights(std::size_t i)
{
    return _weights.at(i);
}

const Matrix &SelfAttention::weights(std::size_t i) const
{
    return _weights.at(i);
}

Matrix &SelfAttention::bias(std::size_t i)
{
    return _bias.at(i);
}

const Matrix &SelfAttention::attention() const noexcept
{
    return _attention;
}

// ---- TokenNorm ---------------------------------------------------------------------------------

namespace {

// Validates the arguments of TokenNorm before the LayerNorm it holds is built.
LayerNorm makeNorm(std::size_t tokens, std::size_t dim, Scalar epsilon)
{
    requirePositive("token_norm", "tokens", tokens);
    requirePositive("token_norm", "dim", dim);
    if (!(epsilon > 0))
        throw Error("token_norm: epsilon must be positive");
    return LayerNorm(dim, epsilon);
}

} // namespace

TokenNorm::TokenNorm(std::size_t tokens, std::size_t dim, Scalar epsilon)
    : _tokens(tokens), _norm(makeNorm(tokens, dim, epsilon))
{
}

Matrix TokenNorm::forward(const Matrix &input, bool training)
{
    requireWidth("token_norm", _tokens * dim(), input.cols());
    const Matrix out = _norm.forward(reshaped(input, input.rows() * _tokens, dim()), training);
    return reshaped(out, input.rows(), _tokens * dim());
}

Matrix TokenNorm::backward(const Matrix &gradOutput)
{
    requireWidth("token_norm", _tokens * dim(), gradOutput.cols());
    const Matrix grad = _norm.backward(reshaped(gradOutput, gradOutput.rows() * _tokens, dim()));
    return reshaped(grad, gradOutput.rows(), _tokens * dim());
}

std::vector<Parameter> TokenNorm::parameters()
{
    return _norm.parameters();
}

std::size_t TokenNorm::outputWidth(std::size_t inputWidth) const
{
    requireWidth("token_norm", _tokens * dim(), inputWidth);
    return inputWidth;
}

std::string TokenNorm::type() const
{
    return "token_norm";
}

void TokenNorm::save(std::ostream &out) const
{
    out << "token_norm tokens=" << _tokens << " dim=" << dim() << " epsilon=" << epsilon() << '\n';
    writeRows(out, _norm.gamma());
    writeRows(out, _norm.beta());
}

std::size_t TokenNorm::tokens() const noexcept
{
    return _tokens;
}

std::size_t TokenNorm::dim() const noexcept
{
    return _norm.features();
}

Scalar TokenNorm::epsilon() const noexcept
{
    return _norm.epsilon();
}

Matrix &TokenNorm::gamma() noexcept
{
    return _norm.gamma();
}

Matrix &TokenNorm::beta() noexcept
{
    return _norm.beta();
}

} // namespace my_torch
