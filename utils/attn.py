import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.toy_block import MyConv, RFBBlock, CFBlock


class ReverseAttention(nn.Module):

    def __init__(self, in_channel, out_channel, depth=3, kernel_size=3, padding=1):
        super(ReverseAttention, self).__init__()
        self.conv_in = MyConv(in_channel, out_channel, 1, is_act=False)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(MyConv(out_channel, out_channel, kernel_size, padding=padding, is_act=False))
        self.conv_out = MyConv(out_channel, 1, 1, is_act=False)

    def forward(self, x, fm):
        fm = F.interpolate(fm, size=x.shape[2:], mode='bilinear', align_corners=False)
        rfm = -1 * (torch.sigmoid(fm)) + 1

        x = rfm.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for mid_conv in self.conv_mid:
            x = F.relu(mid_conv(x), inplace=True)
        out = self.conv_out(x)
        res = out + fm

        return res


class BoundaryAttention(nn.Module):

    def __init__(self):
        super(BoundaryAttention, self).__init__()
        self.rfb2 = RFBBlock(128, 64)
        self.rfb3 = RFBBlock(320, 64)
        self.rfb4 = RFBBlock(512, 64)

        self.cfb = CFBlock(64)

        self.ra2 = ReverseAttention(128, 64, depth=2)
        self.ra3 = ReverseAttention(320, 64, depth=2, kernel_size=5, padding=2)
        self.ra4 = ReverseAttention(512, 64, depth=3, kernel_size=7, padding=3)

    def forward(self, x2, x3, x4):
        # deal with x2,3,4 with receptive field block
        rfb_x2 = self.rfb2(x2)
        rfb_x3 = self.rfb3(x3)
        rfb_x4 = self.rfb4(x4)
        # cascade aggregate 3 rfb_res, get cfb_res
        cfb_res = self.cfb(rfb_x4, rfb_x3, rfb_x2)
        # deal with x2,3,4
        ra4_res = self.ra4(x4, cfb_res)
        ra3_res = self.ra3(x3, ra4_res)
        ra2_res = self.ra2(x2, ra3_res)
        return ra2_res


class ChannelAttention(nn.Module):

    def __init__(self, in_channel=64):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = MyConv(in_channel, in_channel // 16, 1, is_bn=False)
        self.fc2 = MyConv(in_channel // 16, in_channel, 1, is_bn=False)

    def forward(self, x):
        avg_res = self.fc2(self.fc1(self.avg_pool(x)))
        max_res = self.fc2(self.fc1(self.max_pool(x)))
        res = avg_res + max_res
        return torch.sigmoid(res)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()
        self.conv = MyConv(2, 1, kernel_size, is_bn=False)

    def forward(self, x):
        avg_res = torch.mean(x, dim=1, keepdim=True)
        max_res, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_res, max_res], dim=1)
        res = self.conv(res)
        return torch.sigmoid(res)


class StructureAttention(nn.Module):

    def __init__(self):
        super(StructureAttention, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)

    def forward(self, x):
        res = self.sa(self.ca(x) * x) * x
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
