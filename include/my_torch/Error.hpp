#pragma once

#include <stdexcept>

namespace my_torch {

// Any library error. The executables catch it in main and exit with 84.
class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace my_torch
