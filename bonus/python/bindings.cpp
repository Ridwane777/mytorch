// Python module of the my_torch library (ADR-0044), built with pybind11. It only calls the public
// API of include/my_torch/: the library itself is not changed. NumPy arrays are copied into a
// Matrix (and back) in one block; the GIL is released while the library computes.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "my_torch/Error.hpp"
#include "my_torch/Factory.hpp"
#include "my_torch/Network.hpp"

namespace py = pybind11;
using my_torch::Matrix;
using my_torch::Network;
using my_torch::Scalar;
using Array = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;

namespace {

// A 2-D array (rows × columns) as a Matrix; a 1-D array is one row.
Matrix toMatrix(const Array &array)
{
    if (array.ndim() != 1 && array.ndim() != 2)
        throw my_torch::Error("expected a 1-D or 2-D array, got " + std::to_string(array.ndim()) +
                              " dimensions");
    const auto rows = array.ndim() == 2 ? static_cast<std::size_t>(array.shape(0)) : 1;
    const auto cols = static_cast<std::size_t>(array.shape(array.ndim() - 1));
    Matrix matrix(rows, cols);
    std::copy_n(array.data(), rows * cols, matrix.data());
    return matrix;
}

Array toArray(const Matrix &matrix)
{
    Array array({matrix.rows(), matrix.cols()});
    std::copy_n(matrix.data(), matrix.rows() * matrix.cols(), array.mutable_data());
    return array;
}

Array predict(Network &net, const Array &inputs)
{
    const Matrix in = toMatrix(inputs);
    Matrix out;
    {
        py::gil_scoped_release release;
        out = net.predict(in);
    }
    return toArray(out);
}

// Trains with the loss, optimizer, scheduler and settings stored in the network (its "train"
// line, ADR-0013); the keyword arguments override the settings given.
py::list fit(Network &net, const Array &inputs, const Array &targets,
             const std::optional<Array> &validationInputs,
             const std::optional<Array> &validationTargets, std::optional<std::size_t> epochs,
             std::optional<std::size_t> batch, std::optional<std::uint64_t> seed,
             std::optional<std::size_t> patience)
{
    if (validationInputs.has_value() != validationTargets.has_value())
        throw my_torch::Error("validation needs both inputs and targets");
    const my_torch::Dataset train{toMatrix(inputs), toMatrix(targets)};
    std::optional<my_torch::Dataset> validation;
    if (validationInputs)
        validation = my_torch::Dataset{toMatrix(*validationInputs), toMatrix(*validationTargets)};

    const my_torch::TrainingSpec &spec = net.info().training;
    my_torch::TrainConfig config = spec.config;
    config.epochs = epochs.value_or(config.epochs);
    config.batchSize = batch.value_or(config.batchSize);
    config.seed = seed.value_or(config.seed);
    config.patience = patience.value_or(config.patience);
    const auto loss = my_torch::makeLoss(spec.loss);
    const auto optimizer = my_torch::makeOptimizer(spec.optimizer);
    const auto scheduler = my_torch::makeScheduler(spec.scheduler);

    my_torch::History history;
    {
        py::gil_scoped_release release;
        history = net.fit(train, validation ? &*validation : nullptr, *loss, *optimizer, config,
                          scheduler.get());
    }
    py::list out;
    for (const my_torch::EpochStats &s : history) {
        py::dict epoch;
        epoch["train_loss"] = s.trainLoss;
        epoch["train_accuracy"] = s.trainAccuracy;
        epoch["validation_loss"] = s.validationLoss;
        epoch["validation_accuracy"] = s.validationAccuracy;
        epoch["penalty"] = s.penalty;
        out.append(epoch);
    }
    return out;
}

} // namespace

PYBIND11_MODULE(my_torch, m)
{
    m.doc() = "Python bindings of the my_torch neural-network library (ADR-0044)";
    py::register_exception<my_torch::Error>(m, "Error", PyExc_RuntimeError);

    py::class_<Network>(m, "Network")
        // Network is move-only (its layers are unique_ptr), so it is handed over in a unique_ptr.
        .def_static(
            "load",
            [](const std::string &path) { return std::make_unique<Network>(Network::load(path)); },
            py::arg("path"), "Load a network file (binary or text format)")
        .def_static(
            "from_config",
            [](const std::string &path, std::uint64_t index) {
                return std::make_unique<Network>(Network::fromConfig(path, index));
            },
            py::arg("path"), py::arg("index") = 0,
            "Build an untrained network from a configuration file")
        .def("save", &Network::save, py::arg("path"), "Save in the binary format")
        .def("save_text", &Network::saveText, py::arg("path"), "Save in the text format")
        .def("predict", &predict, py::arg("inputs"),
             "Outputs of the network, one row per input row")
        .def("fit", &fit, py::arg("inputs"), py::arg("targets"),
             py::arg("validation_inputs") = py::none(), py::arg("validation_targets") = py::none(),
             py::kw_only(), py::arg("epochs") = py::none(), py::arg("batch") = py::none(),
             py::arg("seed") = py::none(), py::arg("patience") = py::none(),
             "Train with the settings of the network file; returns one dict per epoch")
        .def("summary", &my_torch::modelSummary, "Table of the layers")
        .def_property_readonly("inputs", [](const Network &n) { return n.info().inputs; })
        .def_property_readonly("labels", [](const Network &n) { return n.info().labels; })
        .def_property_readonly("encoding", [](const Network &n) { return n.info().encoding; });
}
