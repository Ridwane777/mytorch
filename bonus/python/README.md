# Python bindings (bonus)

The my_torch library as a Python module, built with pybind11 (ADR-0044): the same compiled code as
the analyzer, called from Python with NumPy arrays. Nothing here is built by the root `make`.

## Build

```sh
python3 -m venv bonus/python/.venv
bonus/python/.venv/bin/pip install pybind11==3.1.0 numpy==2.5.4
make -C bonus python          # builds bonus/python/my_torch.<python version>.so
make -C bonus python_tests    # 6 tests, no dataset needed
```

## Use

```python
import sys
sys.path.insert(0, "bonus/python")
import numpy as np
import my_torch

net = my_torch.Network.load("my_torch_network_final.nn")
print(net.summary(), net.inputs, net.labels)
probabilities = net.predict(np.zeros((2, net.inputs)))   # one row per input row

# Training with the settings of a configuration file (loss, optimizer, train line):
net = my_torch.Network.from_config("configs/final_network.conf")
history = net.fit(x_train, y_train, x_valid, y_valid, epochs=5)   # one dict per epoch
net.save("my_net.nn")          # binary format; net.save_text(...) for the text one
```

Errors of the library are raised as `my_torch.Error` (a `RuntimeError`).

Comparison with the C++ library: [docs/performance/25-python-bindings.md](../../docs/performance/25-python-bindings.md)
(`sh scripts/bench_python_bindings.sh`).
