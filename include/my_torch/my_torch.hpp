#pragma once

// En-tête public du module my_torch : l'analyseur et le générateur n'incluent que lui.

#include "my_torch/Error.hpp"
#include "my_torch/Layer.hpp"
#include "my_torch/Loss.hpp"
#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"
#include "my_torch/Optimizer.hpp"

namespace my_torch {

const char *version() noexcept;

} // namespace my_torch
