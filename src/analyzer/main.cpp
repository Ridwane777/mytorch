#include "Cli.hpp"

#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
    try {
        return analyzer::run({argv + 1, argv + argc});
    } catch (const std::exception &e) {
        std::cerr << "my_torch_analyzer: " << e.what() << std::endl;
        return analyzer::EXIT_ERROR;
    }
}
