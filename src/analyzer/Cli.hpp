#pragma once

#include <string>
#include <vector>

namespace analyzer {

constexpr int EXIT_ERROR = 84;

// Point d'entrée testable de my_torch_analyzer : renvoie le code de sortie.
int run(const std::vector<std::string> &args);

} // namespace analyzer
