#pragma once

#include <stdexcept>

namespace my_torch {

// Toute erreur de la bibliothèque. Les exécutables l'attrapent dans main et quittent avec 84.
class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace my_torch
