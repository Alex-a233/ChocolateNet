import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.toy_block import MyConv


class BoundaryAttention(nn.Module):

    def __init__(self):
        super(BoundaryAttention, self).__init__()

        # self.conv2 = MyConv(128, 32, 1, is_act=False)
        # self.conv3 = MyConv(320, 32, 1, is_act=False)
        # self.conv4 = MyConv(512, 32, 1, is_act=False)
        #
        # self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        #
        # self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        #
        # self.conv5 = MyConv(96, 32, 3, padding=1, is_act=False)

        # v1009
        self.conv2 = MyConv(128, 32, 1, use_bias=True)
        self.conv3 = MyConv(320, 32, 1, use_bias=True)
        self.conv4 = MyConv(512, 32, 1, use_bias=True)

        self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.conv5 = MyConv(96, 32, 3, padding=1, use_bias=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x2, x3, x4):
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 后两个上采样，三个特征图做减法，减出有差异的边界像素
        x3_2 = self.convs3_2(abs(self.up(x3) - x2))  # 2,3层异同点
        x4_2 = self.convs4_2(abs(self.up(self.up(x4)) - x2))  # 2,4层异同点
        x4_3 = self.convs4_3(abs(self.up(x4) - x3))  # 3,4层异同点
        x4_3_2 = self.convs4_3_2(x3_2 + x4_2 + self.up(x4_3))  # 2,3,4层异同点 TODO: -or+?

        o3_2 = self.convm3_2(self.up(x3)) * x2 * x3_2
        o4_2 = self.convm4_2(self.up(self.up(x4))) * x2 * x4_2
        o4_3 = self.convm4_3(self.up(x4)) * x3 * x4_3

        res = torch.cat((self.up(o4_3), o4_2, o3_2), dim=1)
        res = self.conv5(res)
        res = res * x4_3_2 + x2 + self.up(x3) + self.up(self.up(x4))

        return res


class ChannelAttention(nn.Module):

    def __init__(self, in_channel=64):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = MyConv(in_channel, in_channel // 16, 1, is_bn=False, is_act=False)
        self.fc2 = MyConv(in_channel // 16, in_channel, 1, is_bn=False, is_act=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        avg_res = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_res = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        res = avg_res + max_res
        return torch.sigmoid(res)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = MyConv(2, 1, kernel_size, padding=padding, is_bn=False, is_act=False)

    def forward(self, x):
        avg_res = torch.mean(x, dim=1, keepdim=True)
        max_res, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_res, max_res], dim=1)
        res = self.conv(res)
        return torch.sigmoid(res)


class StructureAttention(nn.Module):

    def __init__(self):
        super(StructureAttention, self).__init__()
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.conv = MyConv(64, 32, 1, use_bias=True)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        res = self.conv(x)
        res = F.interpolate(res, scale_factor=0.5, mode='bilinear', align_corners=True)
        return res


if __name__ == '__main__':
    x1 = torch.randn(1, 64, 88, 88)
    x2 = torch.randn(1, 128, 44, 44)
    x3 = torch.randn(1, 320, 22, 22)
    x4 = torch.randn(1, 512, 11, 11)
    ba = BoundaryAttention()
    ba_res = ba(x2, x3, x4)
    print(ba_res.shape)
    sa = StructureAttention()
    sa_res = sa(x1)
    print(sa_res.shape)
