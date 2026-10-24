#pragma once

#include <string>
#include <vector>

namespace generator {

constexpr int EXIT_ERROR = 84;

// Testable entry point of my_torch_generator: returns the exit code. Every configuration is read
// and checked before any network file is written, so a bad argument writes nothing.
int run(const std::vector<std::string> &args);

} // namespace generator
