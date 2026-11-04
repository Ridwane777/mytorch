// my_torch_nn_dump: prints a network file as readable text (ADR-0042). Since network files are
// binary, this is how to look inside one: the output is the text format of ADR-0012, which
// Network::load also reads, so a dumped file can be edited and loaded again.
// Usage: ./my_torch_nn_dump NETWORK [OUTPUT]   (stdout without OUTPUT)
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "my_torch/my_torch.hpp"

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "USAGE\n    ./my_torch_nn_dump NETWORK [OUTPUT]\n\nDESCRIPTION\n"
                     "    Writes NETWORK (binary or text) as text into OUTPUT, or on stdout.\n";
        return 84;
    }
    try {
        const my_torch::Network net = my_torch::Network::load(argv[1]);
        if (argc == 3) {
            net.saveText(argv[2]);
            return 0;
        }
        const std::string temporary = "my_torch_nn_dump_" + std::to_string(getpid()) + ".txt";
        net.saveText(temporary);
        std::cout << std::ifstream(temporary).rdbuf();
        std::remove(temporary.c_str());
    } catch (const std::exception &error) {
        std::cerr << "my_torch_nn_dump: " << error.what() << '\n';
        return 84;
    }
    return 0;
}
