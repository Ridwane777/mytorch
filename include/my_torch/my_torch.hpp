#pragma once

// Public header of the my_torch module: the analyzer and the generator include only this one.

#include "my_torch/Activations.hpp"
#include "my_torch/Dense.hpp"
#include "my_torch/Error.hpp"
#include "my_torch/Layer.hpp"
#include "my_torch/Loss.hpp"
#include "my_torch/Losses.hpp"
#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"
#include "my_torch/Optimizer.hpp"
#include "my_torch/Optimizers.hpp"
#include "my_torch/Schedulers.hpp"

namespace my_torch {

const char *version() noexcept;

} // namespace my_torch
