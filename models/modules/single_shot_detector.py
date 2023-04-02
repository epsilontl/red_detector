import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.functions.prior_box import PriorBox
from models.modules.l2norm import L2Norm
import os


VOC = {
    "num_classes": 21,
    "lr_steps": (80000, 100000, 120000),
    "max_iter": 120000,
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "min_dim": 300,
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "clip": True,
    "name": "VOC",
}

VGG = {"300": [64, 64, "MaxPooling",
               128, 128, "MaxPooling",
               256, 256, 256, "MaxPooling_Ceil",
               512, 512, 512, "MaxPooling",
               512, 512, 512],
       "512": []}


MBOX = {"300": [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        "512": []}


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        in_channels:
        out_channels:
        num_classes: 2
    """

    def __init__(self, in_channels, out_channels, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        # TODO: change cfg into settings.yaml
        self.cfg = VOC
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)  # 8732 prior bounding boxes for VOC(21 classes)

        self._vgg_layer = build_VGGLayer(in_channels=in_channels)
        self.L2Norm = L2Norm(512, 20)
        self._feature_extractor_layer = build_Feature_Extract_Layer(out_channels=out_channels)
        self.vgg_layer, self.feature_extractor_layer = nn.ModuleList(self._vgg_layer), \
                                                       nn.ModuleList(self._feature_extractor_layer)

        self._loc_layer, self._conf_layer = build_MultiBox(self._vgg_layer, self._feature_extractor_layer, num_classes=21)
        self.loc_layer, self.conf_layer = nn.ModuleList(self._loc_layer), nn.ModuleList(self._conf_layer)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch, 3, 300, 300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch, topk, 7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors*4]
                    3: priorbox layers, Shape: [2, num_priors*4]
        """
        architecture, loc, conf = list(), list(), list()
        for idx in range(23):  # apply vgg up to conv4_3 relu
            x = self.vgg_layer[idx](x)
        classifier1 = self.L2Norm(x)
        architecture.append(classifier1)

        for idx in range(23, len(self.vgg_layer)):  # apply vgg up to fc7
            x = self.vgg_layer[idx](x)
        architecture.append(x)   # Classifier2

        for idx, ef_idx in enumerate(self.feature_extractor_layer):  # apply Feature Extract Layer(Classifier3, 4, 5, 6)
            x = F.relu(ef_idx(x), inplace=True)
            if idx % 2 == 1:
                architecture.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(architecture, self.loc_layer, self.conf_layer):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.num_classes),
                  self.priors)

        return output


def build_VGGLayer(in_channels):  # 3 (RGB)
    vgg_layers = list()
    in_channels_ = in_channels
    for item in VGG['300']:
        if item == 'MaxPooling':
            vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif item == 'MaxPooling_Ceil':
            vgg_layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # same as function: ceil/floor
        else:
            vgg_layers += [nn.Conv2d(in_channels_, item, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
            in_channels_ = item
    vgg_layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                   nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(1024, 1024, kernel_size=1),
                   nn.ReLU(inplace=True)]
    return vgg_layers


def build_Feature_Extract_Layer(out_channels):  # 1024
    fe_layers = list()
    fe_layers += [nn.Conv2d(out_channels, 256, kernel_size=1, stride=1),
                  nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                  nn.Conv2d(512, 128, kernel_size=1, stride=1),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                  nn.Conv2d(256, 128, kernel_size=1, stride=1),
                  nn.Conv2d(128, 256, kernel_size=3, stride=1),
                  nn.Conv2d(256, 128, kernel_size=1, stride=1),
                  nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    return fe_layers


def build_MultiBox(vgg_layers, fe_layers, num_classes):
    loc_layers, conf_layers = list(), list()
    vgg_layers_index = [21, 33]

    for idx, vgg_idx in enumerate(vgg_layers_index):
        loc_layers += [nn.Conv2d(in_channels=vgg_layers[vgg_idx].out_channels,   # boxes
                                 out_channels=MBOX['300'][idx]*4,
                                 kernel_size=3, padding=1)]

        conf_layers += [nn.Conv2d(in_channels=vgg_layers[vgg_idx].out_channels,  # classes
                                  out_channels=MBOX['300'][idx]*num_classes,
                                  kernel_size=3, padding=1)]

    for idx, fe_idx in enumerate(fe_layers[1::2], start=2):
        loc_layers += [nn.Conv2d(in_channels=fe_idx.out_channels,
                                 out_channels=MBOX['300'][idx]*4,
                                 kernel_size=3, padding=1)]

        conf_layers += [nn.Conv2d(in_channels=fe_idx.out_channels,
                                  out_channels=MBOX['300'][idx]*num_classes,
                                  kernel_size=3, padding=1)]

    return [loc_layers, conf_layers]
