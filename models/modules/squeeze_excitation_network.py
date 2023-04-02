import torch
import torch.nn as nn
import torch.nn.functional as F


# class SE_Layer(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction):
#         super(SE_Layer, self).__init__()
#
#         self.layer1 = nn.Sequential(nn.BatchNorm2d(in_channels),
#                                     nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(inplace=True))
#
#         self.layer2 = nn.Sequential(nn.BatchNorm2d(64),
#                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(inplace=True))
#
#         self.layer3 = nn.Sequential(nn.BatchNorm2d(64),
#                                     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
#
#         self.layer4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
#
#         self.layer5 = nn.Sequential(nn.Linear(64, out_channels//reduction, bias=False),  # DenserReLU
#                                     nn.ReLU(inplace=True))
#
#         self.layer6 = nn.Sequential(nn.Linear(out_channels//reduction, out_channels, bias=False),  # DenseSigmoid
#                                     nn.Sigmoid())
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         skip_out = out
#         original_out = out
#         # squeeze layer
#         b, c, _, _ = out.size()
#         out = self.layer4(out).view(b, c)
#         out = self.layer5(out)
#         out = self.layer6(out).view(b, c, 1, 1)
#         # excitation layer
#         out = out.expand_as(original_out) * original_out  # layer7
#         out = skip_out + out  # layer8
#         return out


class SE_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super(SE_Layer, self).__init__()

        self.layer1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))

        self.layer4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.layer5 = nn.Sequential(nn.Conv2d(64, out_channels//reduction, kernel_size=1),  # DenserReLU
                                    nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1),  # DenseSigmoid
                                    nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        skip_out = out
        out = self.layer2(out)
        out = self.layer3(out)
        original_out = out
        # squeeze layer
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # excitation layer
        out = out * original_out  # layer7

        out = F.interpolate(out, [skip_out.shape[2], skip_out.shape[3]])
        out = skip_out + out  # layer8

        return out


if __name__ == "__main__":
    net = SE_Layer(64, 256)
    y = net(torch.randn((4, 64, 304, 240)))  # [batch_size, in_chs, H, W]
    print(y.shape)
