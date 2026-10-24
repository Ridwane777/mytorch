#include "Generator.hpp"

#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
    try {
        return generator::run({argv + 1, argv + argc});
    } catch (const std::exception &e) {
        std::cerr << "my_torch_generator: " << e.what() << std::endl;
        return generator::EXIT_ERROR;
    }
}
