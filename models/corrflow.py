import torch
import torch.nn as nn

import math
from .submodule import ResNet18
from .colorizer import Colorizer

import numpy as np

class CorrFlow(nn.Module):
    def __init__(self, args):
        super(CorrFlow, self).__init__()

        # Model options
        self.p = 0.3
        self.feature_extraction = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.colorizer = Colorizer()

    def forward(self, rgb_r, quantized_r, rgb_t):
        feats_r = self.post_convolution(self.feature_extraction(rgb_r))
        feats_t = self.post_convolution(self.feature_extraction(rgb_t))

        quantized_t = self.colorizer(feats_r, feats_t, quantized_r)
        return quantized_t

    def dropout2d(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        if np.random.random() < self.p:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2 + 1), 1))
        drop_ch_ind = np.random.choice(np.arange(3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr