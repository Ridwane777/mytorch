# Using `my_torch` in another project

`my_torch` is a generic neural-network library: it knows nothing about chess. It depends only on
the C++20 standard library, and the analyzer and the generator use it through its public headers,
like any other program would ([architecture](architecture.md)).

## Build and link

```bash
make build/libmy_torch.a                     # the static library, from src/my_torch/
g++ -std=c++20 -O2 -pthread -I<repo>/include my_program.cpp <repo>/build/libmy_torch.a -o my_program
```

One header gives the whole public API: `#include "my_torch/my_torch.hpp"`. Every function reports an
error by throwing `my_torch::Error`.

## A complete example: learning XOR

```cpp
#include <iostream>
#include <memory>

#include "my_torch/my_torch.hpp"

int main()
{
    using namespace my_torch;
    Matrix x(4, 2), y(4, 1); // one row per example
    x(1, 1) = 1; x(2, 0) = 1; x(3, 0) = 1; x(3, 1) = 1;
    y(1, 0) = 1; y(2, 0) = 1;

    Network net;
    net.add(std::make_unique<Dense>(2, 8, 1)); // 2 inputs, 8 neurons, seed 1
    net.add(std::make_unique<Tanh>());
    net.add(std::make_unique<Dense>(8, 1, 2));
    net.add(std::make_unique<Sigmoid>());

    MSE loss;
    Adam adam(0.05);
    TrainConfig config;
    config.epochs = 1000;
    config.batchSize = 4;
    net.fit({x, y}, nullptr, loss, adam, config);

    const Matrix p = net.predict(x);
    for (std::size_t r = 0; r < 4; r++)
        std::cout << x(r, 0) << " xor " << x(r, 1) << " -> " << p(r, 0) << '\n';
}
```

Built outside the repository against `build/libmy_torch.a` only, it printed (2026-10-26):

```
0 xor 0 -> 0.00284834
0 xor 1 -> 0.996561
1 xor 0 -> 0.997854
1 xor 1 -> 0.00263538
```

## What the library offers

| Need | API | Details |
| --- | --- | --- |
| Layers | `Dense` (with L1/L2 and a choice of initialisation), `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout` | ADR-0004, ADR-0010, ADR-0020 |
| Losses | `MSE`, `MAE`, `Huber`, `BinaryCrossEntropy`, `CrossEntropy` (label smoothing, class weights) | ADR-0007 |
| Optimizers and schedules | `SGD`, `Adam`, `AdamW`, `RMSProp`, `Adagrad`; constant, step, exponential, cosine, warm-up | ADR-0008 |
| Training | `Network::fit` with mini-batches, seeded shuffling, validation, early stopping | ADR-0009, ADR-0010 |
| Training stability | Gradient clipping by value (`clipValue`) and by global norm (`clipNorm`), off by default; `fit` stops with an `Error` naming the epoch, the batch and the layer on the first NaN or infinite loss, gradient or weight | ADR-0031 |
| Speed | Matrix products blocked and split over all hardware threads, results identical for any thread count; set `MY_TORCH_THREADS=1` or call `setThreadCount(1)` to stay on one thread | ADR-0024 |
| Splits and cross-validation | `stratifiedSplit` (parts keeping the class proportions), `stratifiedKFold`, `SubsetSource` (train on some rows of a source without copying), `targetClasses` | ADR-0027 |
| Data that does not fit in memory | Implement `DataSource` (`size`, `inputWidth`, `targetWidth`, `fill`) and pass it to `fit`: only one batch is built at a time | ADR-0021 |
| Evaluation | `report` from probabilities or from a confusion matrix (accuracy, precision, recall, F1), `confusion` from probabilities or from class indices, `merge` of classes into groups, `rocAuc`, `calibration`, CSV/JSON export | ADR-0011, ADR-0026 |
| Hyperparameter search | `SearchSpace`, `GridSearch`, `RandomSearch(count, seed)`: the library chooses the combinations, your objective trains and scores each one; a CSV log makes a long search resumable | ADR-0028 |
| Saving | `Network::save`, `Network::load`, `Network::fromConfig` (versioned text file with a checksum) | ADR-0012, ADR-0013 |

## A data source of your own

```cpp
// Rows built on demand, for instance from images on disk or from records in a database.
class MySource : public my_torch::DataSource {
  public:
    std::size_t size() const override { return count; }
    std::size_t inputWidth() const override { return 100; }
    std::size_t targetWidth() const override { return 3; }
    void fill(const std::vector<std::size_t> &rows, my_torch::Matrix &inputs,
              my_torch::Matrix &targets) const override
    {
        // Write example rows[i] into row i; both matrices arrive sized and filled with zeros.
    }
};
```

The chess analyzer does exactly this with `PositionSource` (`src/analyzer/Encoding.hpp`).

## Cross-validation on any source

```cpp
// Five stratified folds of a source; each fold trains a fresh network on a view of its rows.
const std::vector<my_torch::Fold> folds =
    my_torch::stratifiedKFold(my_torch::targetClasses(source), 5, 42);
for (const my_torch::Fold &fold : folds) {
    const my_torch::SubsetSource train(source, fold.train);
    const my_torch::SubsetSource validation(source, fold.validation);
    // net.fit(train, &validation, loss, optimizer, config); then score it
}
```

## Hyperparameter search

```cpp
// Every combination of two hyperparameters; the objective returns a score, higher is better.
my_torch::SearchSpace space;
space.add("lr", {0.0003, 0.001, 0.003});
space.add("hidden", {std::string("256x128"), std::string("512x256")});
const my_torch::SearchResult result = my_torch::GridSearch().run(
    space,
    [&](const my_torch::Trial &t) {
        // build a network with t.text("hidden"), train it with t.number("lr"), score it
        return validationMacroF1;
    },
    "search.csv"); // rerunning skips the trials already in search.csv
const my_torch::TrialResult &best = result.best();
```

`my_torch::RandomSearch(12, 7)` in place of `GridSearch()` tries 12 combinations drawn with seed 7.
The chess network's search is `scripts/hyperparam_search.cpp`.
