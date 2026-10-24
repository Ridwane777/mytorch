#pragma once

#include <string>
#include <vector>

namespace analyzer {

constexpr int EXIT_ERROR = 84;

// Testable entry point of my_torch_analyzer: returns the exit code.
int run(const std::vector<std::string> &args);

} // namespace analyzer
