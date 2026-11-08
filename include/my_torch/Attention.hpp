#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "my_torch/Dense.hpp"
#include "my_torch/Layer.hpp"
#include "my_torch/Normalization.hpp"

namespace my_torch {

// Token layers and self-attention (ADR-0047). A Matrix row is one sample, and a sample made of
// `tokens` tokens of `dim` features is stored flattened token-major: the feature f of token t sits
// in column t·dim + f, so a row is always tokens × dim wide. The layers do not infer the geometry
// from the width of their input: it is given to the constructor (the configuration and the network
// file write it), and a layer throws my_torch::Error when the input it receives has another
// width. A transformer block is built from these layers and the Residual container (ADR-0046).

// Parameter-free reshape of a flat row into tokens (ADR-0047). The input row holds `inputs`
// columns: planes of `tokens` cells (index plane·tokens + cell), except for `count` scalars that
// start at column `scalars`, on a plane boundary. Token t of the output gets one value from every
// plane (the cell t), then the `count` scalars, so it has features() = planes + count features and
// the output is tokens·features() wide. In backward, a cell receives its own gradient and a scalar
// the sum over all the tokens. The layer knows nothing of what the planes hold: the configuration
// gives the layout of its encoding.
class ToTokens : public Layer {
  public:
    // Throws my_torch::Error unless tokens > 0, count > 0, scalars is a multiple of tokens,
    // scalars + count <= inputs, and inputs - count is a positive multiple of tokens.
    ToTokens(std::size_t inputs, std::size_t scalars, std::size_t count, std::size_t tokens);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "to_tokens inputs=… scalars=… count=… tokens=…"
    void save(std::ostream &out) const override;

    std::size_t inputs() const noexcept;
    std::size_t scalars() const noexcept;
    std::size_t count() const noexcept;
    std::size_t tokens() const noexcept;
    // Features per token: the planes, then the scalars.
    std::size_t features() const noexcept;

  private:
    std::size_t _inputs;
    std::size_t _scalars;
    std::size_t _count;
    std::size_t _tokens;
    std::size_t _planes;
};

// A dense layer applied to every token with the same weights: token t of the output is
// token t of the input times W (inputs × outputs), plus the bias b (1 × outputs). The input is
// tokens·inputs wide, the output tokens·outputs wide. Forward and backward view the batch as
// (batch·tokens) × inputs rows and do the products of Dense on it.
class TokenDense : public Layer {
  public:
    // Weights drawn from std::mt19937(seed) following `init` with fan_in = inputs and
    // fan_out = outputs (He normal by default), biases at zero. Throws my_torch::Error on a zero
    // size or a negative regularization coefficient.
    TokenDense(std::size_t tokens, std::size_t inputs, std::size_t outputs, std::uint32_t seed = 42,
               Regularization regularization = {}, Init init = Init::HeNormal);

    Matrix forward(const Matrix &input, bool training) override;
    // dW += Xᵀ·G, db += sum of the rows of G, returns dX = G·Wᵀ, with X and G the (batch·tokens)
    // rows views. Gradients accumulate, as Dense.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // l1·Σ|w| + l2·Σw², as Dense; backward adds the derivative to dW.
    Scalar penalty() const override;
    // "token_dense tokens=… inputs=… outputs=… l1=… l2=…", then one line of weights per input,
    // then the biases.
    void save(std::ostream &out) const override;

    std::size_t tokens() const noexcept;
    Matrix &weights() noexcept;
    const Matrix &weights() const noexcept;
    Matrix &bias() noexcept;
    const Regularization &regularization() const noexcept;

  private:
    std::size_t _tokens;
    Regularization _regularization;
    Matrix _weights;
    Matrix _bias;
    Matrix _weightsGrad;
    Matrix _biasGrad;
    Matrix _input; // (batch·tokens) × inputs view of the last forward
};

// Learned position embedding: adds the table (tokens × dim) to every sample, so the same token
// content at another position gives another vector. The table is drawn from N(0, 0.1²) with the
// seed. Backward sends the gradient through and sums it over the batch for the table.
class Positional : public Layer {
  public:
    // Throws my_torch::Error on a zero size.
    Positional(std::size_t tokens, std::size_t dim, std::uint32_t seed = 42);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "positional tokens=… dim=…", then one line of `dim` numbers per token.
    void save(std::ostream &out) const override;

    std::size_t tokens() const noexcept;
    std::size_t dim() const noexcept;
    Matrix &table() noexcept;
    const Matrix &table() const noexcept;

  private:
    Matrix _table;
    Matrix _tableGrad;
};

// Multi-head self-attention without mask. With X the tokens (tokens × dim) of a sample,
// Q = X·Wq + bq, K = X·Wk + bk, V = X·Wv + bv, split into `heads` slices of dim / heads columns.
// For each head, P = softmax_rows(Q·Kᵀ / sqrt(dim / heads)) and the context is P·V; the heads
// are put side by side and projected: output = context·Wo + bo. Every token reads every token.
//
// Backward is derived by hand: with dC = G·Woᵀ, dV = Pᵀ·dC, dP = dC·Vᵀ,
// dS = P ∘ (dP - rowsum(dP ∘ P)), dQ = dS·K / sqrt(dim / heads), dK = dSᵀ·Q / sqrt(dim / heads),
// and each projection gets dW = Xᵀ·dQ (or dK, dV), db = column sums, dX = sum of the three
// products dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ. The attention matrices are kept by a training forward; after
// a prediction forward, backward recomputes them (input gradients, ADR-0035), which keeps the
// memory of large prediction batches small.
class SelfAttention : public Layer {
  public:
    // Weight matrices drawn from std::mt19937(seed + i) with the Xavier uniform law, biases at
    // zero. Throws my_torch::Error unless tokens, dim and heads are positive and dim is a
    // multiple of heads.
    SelfAttention(std::size_t tokens, std::size_t dim, std::size_t heads, std::uint32_t seed = 42);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "self_attention tokens=… dim=… heads=…", then for the projections q, k, v and o in this
    // order: one line of weights per input feature, then the biases.
    void save(std::ostream &out) const override;

    std::size_t tokens() const noexcept;
    std::size_t dim() const noexcept;
    std::size_t heads() const noexcept;
    // Projection i (0 = q, 1 = k, 2 = v, 3 = o): weights dim × dim, bias 1 × dim.
    Matrix &weights(std::size_t i);
    const Matrix &weights(std::size_t i) const;
    Matrix &bias(std::size_t i);
    const Matrix &bias(std::size_t i) const;
    // Attention matrices of the last training forward: row (sample·heads + head)·tokens + query,
    // column key; every row sums to 1. Empty after a prediction forward.
    const Matrix &attention() const noexcept;

  private:
    std::size_t _tokens;
    std::size_t _dim;
    std::size_t _heads;
    std::array<Matrix, 4> _weights;
    std::array<Matrix, 4> _bias;
    std::array<Matrix, 4> _weightsGrad;
    std::array<Matrix, 4> _biasGrad;
    // Of the last forward, all with (batch·tokens) rows.
    Matrix _input;
    Matrix _q;
    Matrix _k;
    Matrix _v;
    Matrix _context;
    Matrix _attention; // (batch·heads·tokens) × tokens, kept by a training forward only
};

// Layer normalization of each token over its `dim` features (the computation of LayerNorm on
// the (batch·tokens) × dim view), with gamma and beta of `dim` values shared by all tokens.
class TokenNorm : public Layer {
  public:
    // Throws my_torch::Error unless tokens > 0, dim > 0 and epsilon > 0.
    TokenNorm(std::size_t tokens, std::size_t dim, Scalar epsilon = 1e-5);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    // "token_norm tokens=… dim=… epsilon=…", then the lines gamma and beta.
    void save(std::ostream &out) const override;

    std::size_t tokens() const noexcept;
    std::size_t dim() const noexcept;
    Scalar epsilon() const noexcept;
    Matrix &gamma() noexcept;
    Matrix &beta() noexcept;

  private:
    std::size_t _tokens;
    LayerNorm _norm;
};

} // namespace my_torch
