from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map."""
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.resize = cfg.resize
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg.aspect_ratios)
        self.variance = cfg.variance or [0.1]
        self.feature_maps = cfg.feature_maps
        self.min_sizes = cfg.min_sizes
        self.max_sizes = cfg.max_sizes
        self.steps = cfg.steps
        self.aspect_ratios = cfg.aspect_ratios
        self.clip = cfg.clip
        self.gpu = cfg.gpu_device
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # cx = (j + 0.5) * step / min_dim,
                # which means mapping coordinates of feature maps to coordinates of input images
                f_k = self.resize / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.resize
                mean += [cx, cy, s_k, s_k]

                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.resize))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]  # 1:2, 2:1
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]  # 2:1, 3:1
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output.to(self.gpu)
