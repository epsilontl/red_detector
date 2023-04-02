# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.box_utils import match, log_sum_exp
from models.functions.focal_loss import FocalLoss
from models.functions.smooth_l1_loss import Smooth_L1_Loss
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Focal Loss for object confidence.
    Objective Loss:
        L(x,c,l,g) = 1/N * Lloc(x,l,g) + Lconf(x, c)
        Where Lloc is the SmoothL1 Loss weighted by α which is set to 1 by cross val and Lconf is the Focal Loss.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, variance, use_gpu, device):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.variance = variance
        self.use_gpu = use_gpu
        self.device = device
        self.negpos_ratio = 3
        self.focal_loss = FocalLoss(class_num=self.num_classes, alpha=1, gamma=2, size_average=False)
        # self.focal_loss = FocalLoss(alpha=None, gamma=2, size_average=False)
        self.smooth_l1_loss = Smooth_L1_Loss(beta=0.11, reduction="sum")

    def forward(self, predictions, targets):
        """
        MultiBox Loss
        :param predictions: A tuple containing loc_data, conf_data and prior boxes from SSD.
                            loc_data: torch.size(batch_size, num_priors, 4)
                            conf_data: torch.size(batch_size, num_priors, num_classes)
                            priors: torch.size(num_priors, 4)
        :param targets: Ground truth boxes and labels for a batch, [batch_size, num_objs, 5] (last idx is the label)
        """
        batch_loss_l, batch_loss_c, batch_loc_p, batch_N = [], [], [], []
        for idx in range(len(targets)):
            loc_data, conf_data, priors = predictions
            loc_data = loc_data[idx].unsqueeze(0)
            conf_data = conf_data[idx].unsqueeze(0)

            num = loc_data.size(0)
            priors = priors[:loc_data.size(1), :]
            num_priors = (priors.size(0))
            # match priors (default boxes) and ground truth boxes
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.LongTensor(num, num_priors)
            if self.use_gpu:
                loc_t = loc_t.to(self.device)
                conf_t = conf_t.to(self.device)

            if targets[idx].size()[0] == 0:
                batch_loss_l.append(torch.tensor(0.).to(self.device))
                batch_loss_c.append(torch.tensor(0.).to(self.device))
                batch_loc_p.append(torch.tensor([]).to(self.device))
                batch_N.append(torch.tensor(1).to(self.device))
                continue
            truths = targets[idx][:, :-1].data  # gt_box_location
            labels = targets[idx][:, -1].data   # gt_box_cls
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, 0)

            # wrap targets
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)

            pos = conf_t > 0
            num_pos = pos.sum(dim=1, keepdim=True)

            # Localization Loss (Smooth L1)
            # Shape: [batch,num_priors,4]
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = self.smooth_l1_loss(loc_p, loc_t)

            # Confidence Loss (Focal Loss)
            batch_conf_p = conf_data.view(-1, self.num_classes)
            batch_conf_t = conf_t.view(-1)
            loss_c = self.focal_loss(batch_conf_p, batch_conf_t)

            N = torch.clamp(num_pos.data.sum(), min=1)

            loss_l = loss_l / N
            loss_c = loss_c / N
            batch_loss_l.append(loss_l), batch_loss_c.append(loss_c), batch_loc_p.append(loc_p), batch_N.append(N)

        return batch_loss_l, batch_loss_c, batch_loc_p, batch_N
