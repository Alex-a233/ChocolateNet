import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=False,
                 is_bn=True, is_act=True):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)
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

        self.conv_us1 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us2 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us3 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us4 = MyConv(channel, channel, 3, padding=1, is_act=False)
        self.conv_us5 = MyConv(2 * channel, 2 * channel, 3, padding=1, is_act=False)

        self.conv_concat2 = MyConv(2 * channel, 2 * channel, 3, padding=1, is_act=False)
        self.conv_concat3 = MyConv(3 * channel, 3 * channel, 3, padding=1, is_act=False)

        self.conv4 = MyConv(3 * channel, channel, 3, padding=1, is_act=False)

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


class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class FeatureAggregation(nn.Module):

    # def __init__(self, in_channels=64):
    #     super(FeatureAggregation, self).__init__()
    #     self.query_conv = MyConv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, use_bias=True, is_bn=False, is_act=False)
    #     self.key_conv = MyConv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, use_bias=True, is_bn=False, is_act=False)
    #     self.value_conv = MyConv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, is_bn=False, is_act=False)
    #     self.gamma = nn.Parameter(torch.zeros(1))
    #     self.softmax = nn.Softmax(dim=1)
    #     self.scconv = MyConv(in_channels, in_channels // 2, kernel_size=1, is_bn=False, is_act=False)
    #
    # def forward(self, x1, x2):
    #     x = torch.cat([x1, x2], dim=1)
    #     b, c, h, w = x.size()
    #     q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
    #     k = self.key_conv(x).view(b, -1, h * w)
    #     am = torch.bmm(q, k)
    #     a = self.softmax(am)
    #     v = self.value_conv(x).view(b, -1, h * w)
    #     out = torch.bmm(v, a.permute(0, 2, 1))
    #     out = out.view(b, c, h, w)
    #     out = self.gamma * out + x
    #     out = self.scconv(out)
    #     return out

    def __init__(self, num_in=32, num_s=16, mids=4, normalize=False):
        super(FeatureAggregation, self).__init__()
        self.normalize = normalize
        self.num_s = num_s
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_proj = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_extend = MyConv(self.num_s, num_in, kernel_size=1, is_bn=False, is_act=False)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

    def forward(self, x, edge):
        n, c, h, w = x.size()
        edge = F.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # Project and graph reason
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.conv_extend(x_state)

        return out

#     def __init__(self, in_channel=64, out_channel=32):
#         super(FeatureAggregation, self).__init__()
#         self.conva = MyConv(in_channel, out_channel, 3, padding=1, is_act=False)
#         self.dualgcn = DDualGCN(out_channel)
#         self.convb = MyConv(out_channel, out_channel, 3, padding=1, is_act=False)
#         self.bottleneck = MyConv(in_channel + out_channel, out_channel, 3, padding=1, is_act=False)
#
#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim=1)  # (16, 64, 44, 44)
#         output = self.conva(x)  # (16, 32, 44, 44)
#         output = self.dualgcn(output)  # (16, 32, 44, 44)
#         output = self.convb(output)  # (16, 32, 44, 44)
#         output = self.bottleneck(torch.cat([x, output], 1))  # (16, 32, 44, 44)
#         return output
#
#
# class DDualGCN(nn.Module):
#     """
#         Feature GCN with coordinate GCN
#     """
#
#     def __init__(self, in_channel, ratio=4):
#         super(DDualGCN, self).__init__()
#
#         self.local = nn.Sequential(
#             MyConv(in_channel, in_channel, 3, stride=2, padding=1, groups=in_channel, is_act=False),
#             MyConv(in_channel, in_channel, 3, stride=2, padding=1, groups=in_channel, is_act=False),
#             MyConv(in_channel, in_channel, 3, stride=2, padding=1, groups=in_channel, is_act=False)
#         )
#
#         self.gcn_local_attention = SpatialGCN(in_channel)
#         self.phi = MyConv(in_channel, in_channel // ratio * 2, 1, is_act=False)
#         self.theta = MyConv(in_channel, in_channel // ratio, 1, is_act=False)
#
#         #  Interaction Space
#         #  Adjacency Matrix: (-)A_g
#         self.conv_adj = nn.Conv1d(in_channel // ratio, in_channel // ratio, kernel_size=1, bias=False)
#         self.bn_adj = nn.BatchNorm1d(in_channel // ratio)
#
#         #  State Update Function: W_g
#         self.conv_wg = nn.Conv1d(in_channel // ratio * 2, in_channel // ratio * 2, kernel_size=1, bias=False)
#         self.bn_wg = nn.BatchNorm1d(in_channel // ratio * 2)
#
#         #  last fc
#         self.conv3 = MyConv(in_channel // ratio * 2, in_channel, 1, is_act=False)
#         self.alpha = nn.Parameter(torch.zeros(1))
#         self.final = MyConv(in_channel * 2, in_channel, 1, is_act=False)
#
#     def to_matrix(self, x):
#         n, c, h, w = x.size()
#         x = x.view(n, c, -1)
#         return x
#
#     def forward(self, feat):  # (16, 32, 44, 44)
#         # Local
#         x = feat
#         local = self.local(feat)  # (16, 32, 4, 4)
#         local = self.gcn_local_attention(local)  # (16, 32, 4, 4)
#         local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)  # (16, 32, 44, 44)
#         spatial_local_feat = x * local + x  # (16, 32, 44, 44)
#
#         # Projection Space
#         x_sqz, b = x, x
#
#         x_sqz = self.phi(x_sqz)  # (16, 4, 44, 44)
#         x_sqz = self.to_matrix(x_sqz)  # (16, 4, 44*44)
#
#         b = self.theta(b)  # (16, 8, 44, 44)
#         b = self.to_matrix(b)  # (16, 8, 44*44)
#
#         # Project
#         z_idt = torch.matmul(x_sqz, b.transpose(1, 2))  # channel   (16, 4, 8)
#
#         # Interaction Space
#         z = z_idt.transpose(1, 2).contiguous()  # (16, 8, 4)
#         z = self.conv_adj(z)  # (16, 8, 4)
#         z = z.transpose(1, 2).contiguous()  # (16, 4, 8)
#
#         # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
#         z += z_idt  # (16, 4, 8)
#         z = self.conv_wg(z)  # (16, 4, 8)
#         z = self.bn_wg(z)
#
#         # Re-projection Space
#         # Re-project
#         n, _, h, w = x.size()
#         y = torch.matmul(z, b)  # (16, 4, 44*44)
#         y = y.view(n, -1, h, w)  # (16, 4, 44, 44)
#         y = self.conv3(y)  # (16, 32, 44, 44)
#
#         a_out = self.alpha * y + x  # (16, 32, 44, 44)
#
#         # cat or sum, nearly the same results
#         out = self.final(torch.cat((spatial_local_feat, a_out), 1))  # (16, 32, 44, 44)
#         return out
#
#
# class SpatialGCN(nn.Module):
#
#     def __init__(self, in_channel):
#         super(SpatialGCN, self).__init__()
#         out_channel = in_channel // 2
#
#         self.node_k = MyConv(in_channel, out_channel, 1, use_bias=True, is_bn=False, is_act=False)
#         self.node_v = MyConv(in_channel, out_channel, 1, use_bias=True, is_bn=False, is_act=False)
#         self.node_q = MyConv(in_channel, out_channel, 1, use_bias=True, is_bn=False, is_act=False)
#
#         self.conv_wg = nn.Conv1d(out_channel, out_channel, kernel_size=1, bias=False)
#         self.bn_wg = nn.BatchNorm1d(out_channel)
#         self.softmax = nn.Softmax(dim=2)
#
#         self.out = MyConv(out_channel, in_channel, 1, use_bias=True, is_act=False)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):  # (16, 32, 4, 4)
#         node_k = self.node_k(x)  # (16, 16, 4, 4)
#         node_v = self.node_v(x)  # (16, 16, 4, 4)
#         node_q = self.node_q(x)  # (16, 16, 4, 4)
#
#         b, c, h, w = node_k.size()
#
#         node_k = node_k.view(b, c, -1).permute(0, 2, 1)  # (16, 4*4, 16)
#         node_q = node_q.view(b, c, -1)  # (16, 16, 4*4)
#         node_v = node_v.view(b, c, -1).permute(0, 2, 1)  # (16, 4*4, 16)
#
#         AV = torch.bmm(node_q, node_v)  # (16, 16, 16)
#         AV = self.softmax(AV)  # (16, 16, 16)
#         AV = torch.bmm(node_k, AV)  # (16, 4*4, 16)
#         AV = AV.transpose(1, 2).contiguous()  # (16, 16, 4*4)
#         AVW = self.conv_wg(AV)  # (16, 16, 4*4)
#         AVW = self.bn_wg(AVW)
#         AVW = AVW.view(b, c, h, -1)  # (16, 16, 4, 4)
#
#         out = self.gamma * self.out(AVW) + x  # (16, 32, 4, 4)
#         return out
