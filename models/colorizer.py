import torch.nn as nn
import torch.nn.functional as F
import math
from .submodule import one_hot
from spatial_correlation_sampler import SpatialCorrelationSampler

class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        self.D = 4
        self.R = 6  # window size
        self.C = 16
        self.P = self.R * 2 + 1

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

    def prep(self, image):
        _,c,_,_ = image.size()
        x = F.interpolate(image.float(), scale_factor=(1/self.D), mode='bilinear')
        if c == 1:
            x = one_hot(x.long(), self.C)
        return x

    def forward(self, feats_r, feats_t, quantized_r):
        b,c,h,w = quantized_r.size()

        r = self.prep(quantized_r)
        r = F.pad(r, (self.R,self.R,self.R,self.R), mode='replicate')

        corr = self.correlation_sampler(feats_t, feats_r)
        _,_,_,h1,w1 = corr.size()

        corr[corr == 0] = -1e10  # discount padding at edge for softmax
        corr = corr.reshape([b, self.P*self.P, h1*w1])
        corr = F.softmax(corr, dim=1)
        corr = corr.unsqueeze(1)

        image_uf = F.unfold(r, kernel_size=self.P)
        image_uf = image_uf.reshape([b,self.C,self.P*self.P,h1*w1])

        out = (corr * image_uf).sum(2).reshape([b,self.C,h1,w1])

        return out