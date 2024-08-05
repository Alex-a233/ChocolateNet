import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, is_bn=True, is_act=True):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.is_bn = is_bn
        self.is_act = is_act

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_act:
            x = self.relu(x)
        return x


class RFBBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFBBlock, self).__init__()

        self.branch0 = nn.Sequential(
            MyConv(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            MyConv(in_channel, out_channel, 1),
            MyConv(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            MyConv(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            MyConv(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            MyConv(in_channel, out_channel, 1),
            MyConv(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            MyConv(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            MyConv(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            MyConv(in_channel, out_channel, 1),
            MyConv(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            MyConv(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            MyConv(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = MyConv(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = MyConv(in_channel, out_channel, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class CFBlock(nn.Module):

    def __init__(self, channel):
        super(CFBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_us1 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us2 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us3 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us4 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us5 = MyConv(2 * channel, 2 * channel, 3, padding=1, is_act=False)

        self.conv_concat2 = MyConv(2 * channel, 2 * channel, 3, padding=1, is_act=False)
        self.conv_concat3 = MyConv(3 * channel, 3 * channel, 3, padding=1, is_act=False)

        self.conv4 = MyConv(3 * channel, 3 * channel, 3, padding=1, is_act=False)
        self.conv5 = MyConv(3 * channel, 1, 1, is_act=False)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_us1(self.upsample(x1)) * x2
        x3_1 = self.conv_us2(self.upsample(self.upsample(x1))) * self.conv_us3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_us4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_us5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class FeatureAggregation(nn.Module):

    def __init__(self, in_channel=64):
        super(FeatureAggregation, self).__init__()
        self.sconv = MyConv(320, in_channel, 1)
        self.cfb = CFBlock(in_channel)

    def forward(self, x1, x2, x3):
        x1 = self.sconv(x1)
        cfb_res = self.cfb(x1, x2, x3)
        f = F.interpolate(cfb_res, scale_factor=4, mode='bilinear', align_corners=True)

        return f
