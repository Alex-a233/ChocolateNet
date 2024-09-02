import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.toy_block import MyConv


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


class AxialAttention(nn.Module):
    """
    Axial Attention 轴向注意力
    """

    def __init__(self, in_channel, out_channel):
        super(AxialAttention, self).__init__()
        self.conv0 = MyConv(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = MyConv(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.h_attn = SelfAttention(out_channel, mode='h')  # 高度轴向注意力
        self.w_attn = SelfAttention(out_channel, mode='w')  # 宽度轴向注意力

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        hx = self.h_attn(x)  # 求经过两个卷积处理的输入特征 x 的高度轴向注意力
        w_x = self.w_attn(hx)  # 将高度轴向注意力作为输入Tensor求宽度轴向注意力

        return w_x


class SelfAttention(nn.Module):
    """
    自注意力机制允许模型学习并关注输入特征中最相关的部分,
    这在图像识别或语言建模等任务中非常有用,因为这些任务中输入各部分之间的关系很重要。
    :param in_channels: 输入通道数
    :param mode: 模式决定了轴向的维度，高度，宽度或者二者皆备
    """

    def __init__(self, in_channels, mode='hw'):
        super(SelfAttention, self).__init__()
        self.mode = mode
        # Q 一个输入通道数 in_channels，输出通道数 1/8倍 in_channels，大小 1，步长 1，填充 0的卷积
        self.query_conv = MyConv(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        # K 一个输入通道数 in_channels，输出通道数 1/8倍 in_channels，大小 1，步长 1，填充 0的卷积
        self.key_conv = MyConv(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        # V 一个输入通道数 in_channels，输出通道数 in_channels，大小 1，步长 1，填充 0的卷积
        self.value_conv = MyConv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 一个可学习参数 gamma 设定初始值为 0
        self.gamma = nn.Parameter(torch.zeros(1))
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        axis = 1

        if 'h' in self.mode:  # 若mode里含有 h
            axis *= height  # 则 axis 乘高度
        if 'w' in self.mode:  # 若mode里含有 w
            axis *= width  # 则 axis 乘宽度

        view = (batch_size, -1, axis)  # 一个 B, C*W, H 或者 B, C*H, W 或者 B, C, H*W的元组描述 Tensor 的三个维度

        # 将 x 输入 query_conv 处理，然后维度按照 view 进行调整，再将第二，三维度互换
        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        # 将 x 输入 key_conv 处理，然后维度按照 view 进行调整
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)  # 对 pq 和 pk 做矩阵乘法得到注意力图
        attention = self.sigmoid(attention_map)  # 将输出 attention_map 用 sigmoid 激活得到注意力权重

        # 将 x 输入 value_conv 处理，然后维度按照 view 进行调整
        projected_value = self.value_conv(x).view(*view)

        # pv 与 后两个维度互换的 attention 做矩阵乘法
        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        # 将 out 的维度恢复成原始维度
        out = out.view(batch_size, channel, height, width)

        # out 乘以可学习参数 gamma 再加上原始输入Tensor x
        out = self.gamma * out + x
        return out


class BoundaryAttention(nn.Module):

    def __init__(self):
        super(BoundaryAttention, self).__init__()
        # test new ba with no act
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

        # try act
        self.conv2 = MyConv(128, 32, 1, is_act=False)
        self.conv3 = MyConv(320, 32, 1, is_act=False)
        self.conv4 = MyConv(512, 32, 1, is_act=False)

        self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.conv5 = MyConv(96, 32, 3, padding=1, is_act=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x2, x3, x4):
        # test new ba
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 后两个上采样，三个特征图做减法，减出有差异的边界像素
        x3_2 = self.convs3_2(abs(self.up(x3) - x2))  # 2,3层异同点
        x4_2 = self.convs4_2(abs(self.up(self.up(x4)) - x2))  # 2,4层异同点
        x4_3 = self.convs4_3(abs(self.up(x4) - x3))  # 3,4层异同点
        x4_3_2 = self.convs4_3_2(x3_2 + x4_2 + self.up(x4_3))  # 2,3,4层异同点

        o3_2 = self.convm3_2(self.up(x3)) * x2 * x3_2
        o4_2 = self.convm4_2(self.up(self.up(x4))) * x2 * x4_2
        o4_3 = self.convm4_3(self.up(x4)) * x3 * x4_3

        res = torch.cat((self.up(o4_3), o4_2, o3_2), dim=1)
        res = self.conv5(res)
        res *= x4_3_2

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
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.conv = MyConv(64, 32, 1, is_act=False)

    def forward(self, x):
        x = self.ca(x) * x
        res = self.sa(x) * x
        res = self.conv(res)
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
