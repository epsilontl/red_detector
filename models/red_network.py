from models.modules.convlstm_network import ConvLSTM
from models.modules.squeeze_excitation_network import SE_Layer
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.functions.prior_box import PriorBox


class RED(nn.Module):

    def __init__(self, num_classes, in_channels, cfg):
        """
        RED is a recurrent event-based detector, which is composed of
        1). Feature Extractor: SE_Layer + ConvLSTM
        2). Regression Head(dual): SSD
        :param num_classes: 7 classes(for Prophesee_GEN4) + 1 background
        :param in_channels
        :param cfg: configs for prior boxes
        """
        super(RED, self).__init__()
        self.num_classes = num_classes
        self.priorbox = PriorBox(cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())  # 5148 prior bounding boxes for Prophesee(2 classes + 1 bg)
        # feature extraction layer
        self.layer1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.Conv2d(in_channels, out_channels=32, kernel_size=7, stride=2),
                                    nn.ReLU(inplace=True))
        self.layer2 = SE_Layer(in_channels=32, out_channels=64, reduction=16)
        self.layer3 = SE_Layer(in_channels=64, out_channels=64, reduction=16)

        # for input-to-hidden connection
        self.layer4_1 = nn.Sequential(nn.BatchNorm2d(64),
                                      nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1))
        self.layer4_2 = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)

        self.layer5_1 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.layer5_2 = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)

        self.layer6_1 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.layer6_2 = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)

        self.layer7_1 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.layer7_2 = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)

        self.layer8_1 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.layer8_2 = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)
        # dual regression head
        self.loc_layer1 = nn.ModuleList(build_loc_conf_layer(
            in_channels=256, num_classes=self.num_classes, anchor_boxes=cfg.anchor_boxes)[0])
        self.conf_layer1 = nn.ModuleList(build_loc_conf_layer(
            in_channels=256, num_classes=self.num_classes, anchor_boxes=cfg.anchor_boxes)[1])

        self.loc_layer2 = nn.ModuleList(build_loc_conf_layer(
            in_channels=256, num_classes=self.num_classes, anchor_boxes=cfg.anchor_boxes)[0])
        self.conf_layer2 = nn.ModuleList(build_loc_conf_layer(
            in_channels=256, num_classes=self.num_classes, anchor_boxes=cfg.anchor_boxes)[1])

    def forward(self, x, prev_states):
        """
        :param x: batch_size x input_channels x 512 x 512
        :param prev_states: previous LSTM states for every layer
        :return: (tuple) loc: [batch, num_priors, coords]
                         conf: [batch, num_priors, confidence]
                         prior: [batch, num_priors]
        """
        blocks, states, loc1, conf1, loc2, conf2 = list(), list(), list(), list(), list(), list()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # 64

        x = self.layer4_1(x)
        if prev_states is None:
            prev_states = [None] * 5  # 5 ConvLSTM layers
        state1 = self.layer4_2(x, prev_states[0])  # hidden, ceil
        x = state1[0]
        blocks.append(x), states.append(list(state1))

        x = self.layer5_1(x)  # 32
        state2 = self.layer5_2(x, prev_states[1])
        x = state2[0]
        blocks.append(x), states.append(list(state2))

        x = self.layer6_1(x)  # 16
        state3 = self.layer6_2(x, prev_states[2])
        x = state3[0]
        blocks.append(x), states.append(list(state3))

        x = self.layer7_1(x)  # 8
        state4 = self.layer7_2(x, prev_states[3])
        x = state4[0]
        blocks.append(x), states.append(list(state4))

        x = self.layer8_1(x)  # 4
        state5 = self.layer8_2(x, prev_states[4])
        x = state5[0]
        blocks.append(x), states.append(list(state5))

        for (x, l1, c1, l2, c2) in zip(blocks, self.loc_layer1, self.conf_layer1, self.loc_layer2, self.conf_layer2):
            loc1.append(l1(x).permute(0, 2, 3, 1).contiguous())   # [batch_size, out_channels, H, W]
            conf1.append(c1(x).permute(0, 2, 3, 1).contiguous())  # o.size(0) = 1
            loc2.append(l2(x).permute(0, 2, 3, 1).contiguous())
            conf2.append(c2(x).permute(0, 2, 3, 1).contiguous())

        loc1 = torch.cat([o.view(o.size(0), -1) for o in loc1], 1)
        conf1 = torch.cat([o.view(o.size(0), -1) for o in conf1], 1)
        loc2 = torch.cat([o.view(o.size(0), -1) for o in loc2], 1)
        conf2 = torch.cat([o.view(o.size(0), -1) for o in conf2], 1)

        out1 = (loc1.view(loc1.size(0), -1, 4),  # loc_preds (batch, M, 4)
                conf1.view(conf1.size(0), -1, self.num_classes),  # conf_preds (batch, M, num_classes)
                self.priors)   # default boxes (M, 4)
        out2 = (loc2.view(loc2.size(0), -1, 4),
                conf2.view(conf2.size(0), -1, self.num_classes),
                self.priors)
        return out1, out2, states


def build_loc_conf_layer(in_channels, num_classes, anchor_boxes):  # 256
    loc_layer, conf_layer = list(), list()
    # number of boxes per voxel
    for i in range(len(anchor_boxes)):  # 5 ConvLSTM
        # box_coords * num_boxes_per_voxel
        loc_layer += [nn.Conv2d(in_channels=in_channels, out_channels=4*anchor_boxes[i], kernel_size=3, padding=1)]
        # num_classes * num_boxes_per_voxel
        conf_layer += [nn.Conv2d(in_channels=in_channels, out_channels=num_classes*anchor_boxes[i],
                                 kernel_size=3, padding=1)]
    return [loc_layer, conf_layer]

