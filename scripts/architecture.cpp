// Architecture of a network, layer by layer (docs/performance/20-cnn.md, ADR-0037).
// Usage: architecture NETWORK_OR_CONFIG
// NETWORK_OR_CONFIG is a network file (.nn) or a configuration (.conf). Prints one TSV line per
// layer, with a header line:
//   index  type  settings  output  parameters
// index 0 is the input. `settings` is the first line the layer writes in the network file
// (Layer::save) without its type, `output` the shape after the layer ("19x8x8" for planes, the
// width otherwise), `parameters` the number of trained values, counted from Layer::parameters()
// as modelSummary does; the widths come from Layer::outputWidth, like modelSummary. The last
// line is "total" with the sum.
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "my_torch/my_torch.hpp"

namespace {

using my_torch::PlaneShape;

std::string shapeText(const PlaneShape &shape)
{
    return std::to_string(shape.channels) + "x" + std::to_string(shape.height) + "x" +
           std::to_string(shape.width);
}

// Planes produced by the layer, if it is one of the layers that give a shape.
std::optional<PlaneShape> planesOf(const my_torch::Layer &layer)
{
    if (const auto *l = dynamic_cast<const my_torch::ToPlanes *>(&layer))
        return l->outputShape();
    if (const auto *l = dynamic_cast<const my_torch::Conv2D *>(&layer))
        return l->outputShape();
    if (const auto *l = dynamic_cast<const my_torch::MaxPool2D *>(&layer))
        return l->outputShape();
    if (const auto *l = dynamic_cast<const my_torch::AvgPool2D *>(&layer))
        return l->outputShape();
    return std::nullopt;
}

std::string settingsOf(const my_torch::Layer &layer)
{
    std::ostringstream saved;
    layer.save(saved);
    std::istringstream in(saved.str());
    std::string line;
    std::getline(in, line);
    const std::size_t space = line.find(' ');
    return space == std::string::npos ? "" : line.substr(space + 1);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: architecture NETWORK_OR_CONFIG\n";
        return 1;
    }
    try {
        const std::string path = argv[1];
        const bool config = path.size() > 5 && path.compare(path.size() - 5, 5, ".conf") == 0;
        my_torch::Network net =
            config ? my_torch::Network::fromConfig(path) : my_torch::Network::load(path);
        std::size_t width = net.info().inputs;
        std::optional<PlaneShape> shape;
        std::size_t total = 0;

        std::cout << "index\ttype\tsettings\toutput\tparameters\n";
        std::cout << "0\tinput\t" << net.info().encoding << "\t" << width << "\t0\n";
        for (std::size_t i = 0; i < net.layers().size(); i++) {
            my_torch::Layer &layer = *net.layers()[i];
            std::size_t count = 0;
            for (const my_torch::Parameter &p : layer.parameters())
                count += p.value->size();
            const std::size_t next = layer.outputWidth(width);
            if (const auto planes = planesOf(layer))
                shape = planes;
            else if (!shape || next != width) // dense, global pooling: a flat width
                shape.reset();
            width = next;
            total += count;
            std::cout << i + 1 << '\t' << layer.type() << '\t' << settingsOf(layer) << '\t'
                      << (shape ? shapeText(*shape) : std::to_string(width)) << '\t' << count
                      << '\n';
        }
        std::cout << "total\t\t\t" << width << '\t' << total << '\n';
    } catch (const my_torch::Error &error) {
        std::cerr << "architecture: " << error.what() << '\n';
        return 84;
    }
    return 0;
}
