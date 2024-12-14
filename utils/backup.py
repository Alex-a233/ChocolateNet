import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_block import MyConv


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


class RFBBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFBBlock, self).__init__()

        self.branch0 = nn.Sequential(
            MyConv(in_channel, out_channel, 1, is_act=False),
        )
        self.branch1 = nn.Sequential(
            MyConv(in_channel, out_channel, 1, is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1), is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), is_act=False),
            MyConv(out_channel, out_channel, 3, padding=3, dilation=3, is_act=False)
        )
        self.branch2 = nn.Sequential(
            MyConv(in_channel, out_channel, 1, is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2), is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0), is_act=False),
            MyConv(out_channel, out_channel, 3, padding=5, dilation=5, is_act=False)
        )
        self.branch3 = nn.Sequential(
            MyConv(in_channel, out_channel, 1, is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3), is_act=False),
            MyConv(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0), is_act=False),
            MyConv(out_channel, out_channel, 3, padding=7, dilation=7, is_act=False)
        )

        self.conv_cat = MyConv(4 * out_channel, out_channel, 3, padding=1, is_act=False)
        self.conv_res = MyConv(in_channel, out_channel, 1, is_act=False)
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
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_us1 = MyConv(channel, channel, 3, padding=1, use_bias=True, is_act=False)
        self.conv_us2 = MyConv(channel, channel, 3, padding=1, use_bias=True, is_act=False)
        self.conv_us3 = MyConv(channel, channel, 3, padding=1, use_bias=True, is_act=False)
        self.conv_us4 = MyConv(channel, channel, 3, padding=1, use_bias=True, is_act=False)
        self.conv_us5 = MyConv(2 * channel, 2 * channel, 3, padding=1, use_bias=True, is_act=False)

        self.conv_concat2 = MyConv(2 * channel, 2 * channel, 3, padding=1, use_bias=True, is_act=False)
        self.conv_concat3 = MyConv(3 * channel, 3 * channel, 3, padding=1, use_bias=True, is_act=False)

        self.conv4 = MyConv(3 * channel, channel, 3, padding=1, use_bias=True, is_act=False)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_us1(self.ups(x1)) * x2
        x3_1 = self.conv_us2(self.ups(self.ups(x1))) * self.conv_us3(self.ups(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_us4(self.ups(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_us5(self.ups(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)

        return x
