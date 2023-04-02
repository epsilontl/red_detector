import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    This criterion is a implementation of Focal Loss, which is proposed in Focal Loss for Dense Object Detection.
    Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each mini_batch.

    :param class_num:
    :param alpha: (1D Tensor, Variable) the scalar factor for this criterion
    :param gamma: (float, double) gamma > 0; reduces the relative loss for well-classified examples (p > 0.5),
                                             putting more focus on hard, mis-classified examples
    :param size_average(bool): By default, the losses are averaged over observations for each mini_batch.
                               However, if the field size_average is set to False, the losses are instead summed for
                               each mini_batch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, list):
            assert len(alpha) == class_num
        else:
            # assert alpha < 1
            self.alpha = torch.zeros(class_num)
            self.alpha += (1 - alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, preds, labels):
        """
        :param preds: (tensor) [N, num_classes]
        :param labels: (tensor) [N]
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_softmax = torch.clamp(preds_softmax, max=0.999, min=0.001)  # avoid nan
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.long().view(-1, 1))

        preds_logsoft = preds_logsoft.gather(1, labels.long().view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.long().view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        # loss = torch.mul(self.alpha, loss.t())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# class FocalLoss(nn.Module):
#     """
#     This criterion is a implementation of Focal Loss, which is proposed in Focal Loss for Dense Object Detection.
#     Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#     The losses are averaged across observations for each mini_batch.
#
#     :param class_num:
#     :param alpha: (1D Tensor, Variable) the scalar factor for this criterion
#     :param gamma: (float, double) gamma > 0; reduces the relative loss for well-classified examples (p > 0.5),
#                                              putting more focus on hard, mis-classified examples
#     :param size_average(bool): By default, the losses are averaged over observations for each mini_batch.
#                                However, if the field size_average is set to False, the losses are instead summed for
#                                each mini_batch.
#     """
#     def __init__(self, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         """
#         :param preds: (tensor) [N, num_classes]
#         :param labels: (tensor) [N]
#         """
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)
#             input = input.transpose(1, 2)
#             input = input.contiguout().view(-1, input.size(2))
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma*logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


if __name__ == '__main__':
    focal_loss = FocalLoss(class_num=2, alpha=0.25, gamma=2, size_average=True)
    inputs = torch.rand([4, 2])  # (N, C)
    labels = torch.ones(4)  # N
    print(inputs, '\n', labels)
    loss = focal_loss(inputs, labels)
    print(loss)
