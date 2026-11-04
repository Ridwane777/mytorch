"""PyTorch twin of a my_torch network configuration (#42, ADR-0040).

The architecture is read from the same configuration file as the my_torch network
(configs/final_network.conf, configs/cnn_network.conf): nothing is typed twice. The final softmax
is left to the loss (torch.nn.functional.cross_entropy works on the scores), which is the same
function as my_torch's fused softmax + cross-entropy (ADR-0007).

Layout conventions, from include/my_torch/Dense.hpp and Convolution.hpp:
  - dense: my_torch stores W as inputs x neurons, torch.nn.Linear as neurons x inputs (transposed);
  - conv2d: my_torch stores filters x (channels * k * k), channel-major, i.e. torch's
    (filters, channels, k, k) flattened;
  - a sample is a flat row, images are channel-major (channels x height x width), like torch's NCHW.

This file is used only under comparison/ (ADR-0036).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclasses.dataclass
class Config:
    """What the training needs from a configuration file."""

    encoding: str
    inputs: int
    labels: list[str]
    loss: str
    optimizer: str
    optimizer_params: dict[str, str]
    train: dict[str, str]
    init_seed: int
    layers: list[tuple[str, dict[str, str], list[str]]]  # (type, key=value settings, positional)


def _split(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    named = dict(t.split("=", 1) for t in tokens if "=" in t)
    positional = [t for t in tokens if "=" not in t]
    return named, positional


def read_config(path: str | Path) -> Config:
    """Parses the configuration syntax of Network::fromConfig (ADR-0013)."""
    lines = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    cfg: dict[str, object] = {"init_seed": 42, "train": {}, "optimizer_params": {}}
    layers: list[tuple[str, dict[str, str], list[str]]] = []
    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        key, rest = tokens[0], tokens[1:]
        i += 1
        if key == "input":
            cfg["encoding"], cfg["inputs"] = rest[0], int(rest[1])
        elif key == "labels":
            n = int(rest[0])
            cfg["labels"] = lines[i : i + n]
            i += n
        elif key == "loss":
            cfg["loss"] = rest[0]
        elif key == "optimizer":
            cfg["optimizer"] = rest[0]
            cfg["optimizer_params"], _ = _split(rest[1:])
        elif key == "scheduler":
            if rest[0] != "none":
                raise ValueError("only 'scheduler none' is reproduced in PyTorch")
        elif key == "train":
            cfg["train"], _ = _split(rest)
        elif key == "init":
            cfg["init_seed"] = int(_split(rest)[0]["seed"])
        elif key == "layers":
            n = int(rest[0])
            for line in lines[i : i + n]:
                t = line.split()
                named, positional = _split(t[1:])
                layers.append((t[0], named, positional))
            i += n
        else:
            raise ValueError(f"unknown line in {path}: {lines[i - 1]!r}")
    cfg["layers"] = layers
    return Config(**cfg)  # type: ignore[arg-type]


class BoardPlanes(nn.Module):
    """board_planes (include/my_torch/Convolution.hpp): 901 inputs -> 19 planes of 8 x 8.

    The square planes (64 columns each) keep their order, skipping the `count` scalars that start
    at column `scalars`; each scalar then becomes one more plane, repeated on the 64 squares.
    """

    def __init__(self, inputs: int, scalars: int, count: int):
        super().__init__()
        self.inputs, self.scalars, self.count = inputs, scalars, count
        self.square_planes = (inputs - count) // 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squares = torch.cat([x[:, : self.scalars], x[:, self.scalars + self.count :]], dim=1)
        scalars = x[:, self.scalars : self.scalars + self.count]
        planes = torch.cat(
            [squares.reshape(-1, self.square_planes, 64), scalars.unsqueeze(2).expand(-1, -1, 64)],
            dim=1,
        )
        return planes.reshape(x.shape[0], -1)


class Image(nn.Module):
    """A flat channel-major row seen as channels x height x width, around one layer."""

    def __init__(self, layer: nn.Module, channels: int, height: int, width: int):
        super().__init__()
        self.layer, self.shape = layer, (channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.reshape(x.shape[0], *self.shape)).reshape(x.shape[0], -1)


@dataclasses.dataclass
class Entry:
    """A trained parameter and its my_torch name, in the order of Network::parameters()."""

    layer: int
    kind: str  # "dense", "conv2d"
    index: int  # 0 = weights, 1 = bias
    tensor: torch.nn.Parameter

    def to_mytorch(self, tensor: torch.Tensor) -> np.ndarray:
        """A tensor shaped like this parameter, in my_torch's rows x cols layout."""
        a = tensor.detach().cpu().double().numpy()
        if self.index == 1:
            return a.reshape(1, -1)
        if self.kind == "dense":
            return a.T.copy()
        return a.reshape(a.shape[0], -1)

    def from_mytorch(self, array: np.ndarray) -> torch.Tensor:
        if self.index == 1:
            return torch.from_numpy(array.reshape(self.tensor.shape))
        if self.kind == "dense":
            return torch.from_numpy(array.T.copy())
        return torch.from_numpy(array.reshape(self.tensor.shape))


class Net(nn.Module):
    """The layers of a configuration, scores out (no final softmax)."""

    def __init__(self, config: Config):
        super().__init__()
        modules: list[nn.Module] = []
        self.entries: list[Entry] = []
        types = [t for t, _, _ in config.layers]
        if types[-1] != "softmax" or config.loss != "cross_entropy":
            raise ValueError("expected a final softmax and the cross_entropy loss")
        for i, (kind, named, positional) in enumerate(config.layers[:-1]):
            if kind == "dense":
                layer: nn.Module = nn.Linear(int(positional[0]), int(positional[1]))
                self._register(i, "dense", layer)
            elif kind == "relu":
                layer = nn.ReLU()
            elif kind == "board_planes":
                layer = BoardPlanes(int(named["inputs"]), int(named["scalars"]), int(named["count"]))
            elif kind == "conv2d":
                conv = nn.Conv2d(
                    int(named["channels"]),
                    int(named["filters"]),
                    int(named["kernel"]),
                    stride=int(named["stride"]),
                    padding=int(named["padding"]),
                )
                self._register(i, "conv2d", conv)
                layer = Image(conv, int(named["channels"]), int(named["height"]), int(named["width"]))
            elif kind == "max_pool2d":
                layer = Image(
                    nn.MaxPool2d(int(named["size"]), stride=int(named["stride"])),
                    int(named["channels"]),
                    int(named["height"]),
                    int(named["width"]),
                )
            else:
                raise ValueError(f"layer '{kind}' is not reproduced in PyTorch")
            modules.append(layer)
        self.layers = nn.Sequential(*modules)

    def _register(self, index: int, kind: str, layer: nn.Module) -> None:
        self.entries.append(Entry(index, kind, 0, layer.weight))
        self.entries.append(Entry(index, kind, 1, layer.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def read_manifest(path: str | Path) -> list[tuple[int, str, int, int, int]]:
    """(layer, type, index, rows, cols) for each parameter, from scripts/pytorch_reference.cpp."""
    out = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            layer, kind, index, rows, cols = line.split()
            out.append((int(layer), kind, int(index), int(rows), int(cols)))
    return out


def read_parameters(path: str | Path, manifest) -> list[np.ndarray]:
    """The binary float64 file of pytorch_reference.cpp as one (rows, cols) array per parameter."""
    flat = np.fromfile(path, dtype="<f8")
    arrays, offset = [], 0
    for _, _, _, rows, cols in manifest:
        arrays.append(flat[offset : offset + rows * cols].reshape(rows, cols))
        offset += rows * cols
    if offset != flat.size:
        raise ValueError(f"{path}: {flat.size} values, the manifest describes {offset}")
    return arrays


def check_manifest(net: Net, manifest) -> None:
    """The parameters of the torch network are the ones my_torch wrote, one for one."""
    if len(manifest) != len(net.entries):
        raise ValueError(f"{len(manifest)} parameters in my_torch, {len(net.entries)} in PyTorch")
    for entry, (layer, kind, index, rows, cols) in zip(net.entries, manifest):
        shape = entry.to_mytorch(entry.tensor).shape
        if (entry.layer, entry.kind, entry.index, *shape) != (layer, kind, index, rows, cols):
            raise ValueError(f"layer {layer} {kind}[{index}]: {rows}x{cols} in my_torch, {shape} here")


def load_parameters(net: Net, arrays: list[np.ndarray]) -> None:
    """Copies my_torch's weights into the torch network (in its dtype)."""
    with torch.no_grad():
        for entry, array in zip(net.entries, arrays):
            entry.tensor.copy_(entry.from_mytorch(array))


def read_bytes(prefix: str | Path, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The inputs (rows x width, bytes 0/1) and classes written by `pytorch_reference export`."""
    x = np.fromfile(f"{prefix}.x.u8", dtype=np.uint8)
    y = np.fromfile(f"{prefix}.y.u8", dtype=np.uint8)
    return x.reshape(-1, width), y
