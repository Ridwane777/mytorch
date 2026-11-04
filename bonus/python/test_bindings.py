"""Tests of the Python module (ADR-0044), with no dataset: run by `make -C bonus python_tests`."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import my_torch

ROOT = Path(__file__).resolve().parents[2]
PUSHED = ROOT / "my_torch_network_final.nn"

# Two inputs, XOR of their signs; small enough to learn in a few hundred epochs.
CONFIG = """input raw 2
labels 2
no
yes

loss cross_entropy
optimizer adam lr=0.01
scheduler none
train epochs=300 batch=8 seed=42
init seed=1

layers 4
dense 2 8
tanh
dense 8 2
softmax
"""


def xor_data():
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, size=(64, 2))
    classes = (inputs[:, 0] * inputs[:, 1] > 0).astype(int)
    return inputs, np.eye(2)[classes]


class Bindings(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.config = Path(self.dir.name) / "xor.conf"
        self.config.write_text(CONFIG)

    def tearDown(self):
        self.dir.cleanup()

    def test_pushed_network_loads(self):
        net = my_torch.Network.load(str(PUSHED))
        self.assertEqual(net.inputs, 901)
        self.assertEqual(net.labels[0], "Nothing")
        probabilities = net.predict(np.zeros((3, 901)))
        self.assertEqual(probabilities.shape, (3, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_one_row_as_a_vector(self):
        net = my_torch.Network.from_config(str(self.config))
        self.assertEqual(net.predict(np.array([0.5, -0.5])).shape, (1, 2))

    def test_fit_learns(self):
        net = my_torch.Network.from_config(str(self.config))
        inputs, targets = xor_data()
        history = net.fit(inputs, targets)
        self.assertEqual(len(history), 300)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"] / 2)
        self.assertGreater(history[-1]["train_accuracy"], 0.9)

    def test_fit_overrides_and_validation(self):
        net = my_torch.Network.from_config(str(self.config))
        inputs, targets = xor_data()
        history = net.fit(inputs[:48], targets[:48], inputs[48:], targets[48:], epochs=5)
        self.assertEqual(len(history), 5)
        self.assertGreater(history[-1]["validation_loss"], 0)

    def test_save_and_load_give_the_same_outputs(self):
        net = my_torch.Network.from_config(str(self.config))
        inputs, _ = xor_data()
        expected = net.predict(inputs)
        for save in ("save", "save_text"):
            path = Path(self.dir.name) / f"{save}.nn"
            getattr(net, save)(str(path))
            np.testing.assert_array_equal(my_torch.Network.load(str(path)).predict(inputs),
                                          expected)

    def test_library_errors_become_python_errors(self):
        with self.assertRaises(my_torch.Error):
            my_torch.Network.load(str(Path(self.dir.name) / "missing.nn"))
        net = my_torch.Network.from_config(str(self.config))
        with self.assertRaises(my_torch.Error):
            net.predict(np.zeros((2, 3)))
        with self.assertRaises(my_torch.Error):
            net.predict(np.zeros((2, 2, 2)))
        self.assertTrue(issubclass(my_torch.Error, RuntimeError))


if __name__ == "__main__":
    unittest.main()
