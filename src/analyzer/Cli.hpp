#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace analyzer {

constexpr int EXIT_ERROR = 84;

// Positions encoded and predicted at a time in --predict, so memory stays bounded on large files.
constexpr std::size_t PREDICT_CHUNK = 4096;

// Testable entry point of my_torch_analyzer: returns the exit code, throws my_torch::Error on any
// error (main prints it and exits 84). Every input is read and checked before anything is printed
// or written.
int run(const std::vector<std::string> &args);

// Index of the network label matching an expected output: equal, or equal to its first word, so
// that "Check White" trains the "Check" output of a network without colors. Throws if none
// matches.
std::size_t labelIndex(const std::string &expected, const std::vector<std::string> &labels);

} // namespace analyzer
