#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "my_torch/Layer.hpp"

namespace my_torch {

using LayerSequence = std::vector<std::unique_ptr<Layer>>;

// Width of the rows a sequence of layers gives for rows of `inputWidth` columns, checking every
// layer on the way (Layer::outputWidth, and the features of the normalization layers). Throws
// my_torch::Error naming the layer, counted from 1 within the sequence.
std::size_t sequenceWidth(const LayerSequence &layers, std::size_t inputWidth);

// Residual block (ADR-0046): output = F(x) + x, where F is the sequence of layers it holds. The
// sequence must give back the width it receives (identity shortcut, no projection). backward
// sends the gradient through F and adds it once more for the shortcut.
class Residual : public Layer {
  public:
    // Throws my_torch::Error on an empty sequence.
    explicit Residual(LayerSequence layers);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    Matrix forwardMonteCarlo(const Matrix &input) override;
    std::vector<Parameter> parameters() override;
    Scalar penalty() const override;
    // "residual layers=N", then the records of its N layers.
    void save(std::ostream &out) const override;
    void saveState(std::ostream &out) const override;
    void loadState(std::istream &in) override;
    // Throws my_torch::Error unless the sequence gives back inputWidth.
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    std::vector<Layer *> children() const override;

  private:
    LayerSequence _layers;
};

// Branches side by side (ADR-0046): every branch reads the same input, and the output is their
// outputs one after the other (branch 0's columns, then branch 1's…). backward cuts the gradient
// into one slice per branch and sums the input gradients of the branches.
class Concat : public Layer {
  public:
    // Throws my_torch::Error with fewer than two branches or an empty branch.
    explicit Concat(std::vector<LayerSequence> branches);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    Matrix forwardMonteCarlo(const Matrix &input) override;
    std::vector<Parameter> parameters() override;
    Scalar penalty() const override;
    // "concat branches=N", then for each branch "branch layers=M" and the records of its layers.
    void save(std::ostream &out) const override;
    void saveState(std::ostream &out) const override;
    void loadState(std::istream &in) override;
    std::size_t outputWidth(std::size_t inputWidth) const override;
    std::string type() const override;
    std::vector<Layer *> children() const override;

  private:
    Matrix join(const std::vector<Matrix> &outputs);

    std::vector<LayerSequence> _branches;
    std::vector<std::size_t> _widths; // output width of each branch at the last forward
};

} // namespace my_torch
