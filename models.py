"""
Model registry for Stage 1. Every model maps (batch, leads, samples) -> (batch, n_out) logits.
Step 2 uses only resnet1d_wang; the step-4 candidates get added here later.
"""
from functools import partial

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Three convolutions (kernels 8, 5, 3) with batch norm and a residual shortcut."""

    def __init__(self, cin, cout, kernels=(8, 5, 3)):
        super().__init__()
        layers, c = [], cin
        for i, k in enumerate(kernels):
            layers += [nn.Conv1d(c, cout, k, padding="same", bias=False), nn.BatchNorm1d(cout)]
            if i < len(kernels) - 1:
                layers.append(nn.ReLU(inplace=True))
            c = cout
        self.body = nn.Sequential(*layers)
        self.skip = (nn.Identity() if cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ResNet1dWang(nn.Module):
    """Wang et al. (2017) time-series ResNet, one of the top PTB-XL benchmark models.

    Three residual blocks (64, 128, 128 channels), global average pooling, linear head.
    Global pooling makes it accept any input length, which the window-length tests need.

    stem_stride > 1 adds a learned strided convolution in front of the blocks. It
    shortens the sequence by that factor (5 turns 500 Hz into a 100 Hz-rate feature
    sequence), so each feature covers more time and computation drops about 5x,
    while the filters still see the raw 500 Hz signal.
    """

    def __init__(self, in_ch, n_out, width=64, stem_stride=1):
        super().__init__()
        if stem_stride > 1:
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, width, 2 * stem_stride, stride=stem_stride, bias=False),
                nn.BatchNorm1d(width), nn.ReLU(inplace=True))
            first_in = width
        else:
            self.stem = nn.Identity()
            first_in = in_ch
        self.blocks = nn.Sequential(
            ResBlock(first_in, width), ResBlock(width, 2 * width), ResBlock(2 * width, 2 * width))
        self.head = nn.Linear(2 * width, n_out)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=-1))


MODELS = {
    "resnet1d_wang": ResNet1dWang,
    "resnet1d_wang_s5": partial(ResNet1dWang, stem_stride=5),
}


def build_model(name, in_ch, n_out, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'; available: {list(MODELS)}")
    return MODELS[name](in_ch, n_out, **kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
