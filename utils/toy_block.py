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
        self.act = nn.ReLU(inplace=True)
        self.is_bn = is_bn
        self.is_act = is_act

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_act:
            x = self.act(x)
        return x


class GCN1(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN1, self).__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(in_features, hidden_features))
        nn.init.xavier_uniform_(self.weight1)
        self.weight2 = nn.Parameter(torch.FloatTensor(hidden_features, out_features))
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, x, adj):
        support1 = torch.matmul(x, self.weight1)
        output1 = F.relu(torch.matmul(adj, support1))
        support2 = torch.matmul(output1, self.weight2)
        output2 = torch.matmul(adj, support2)

        return output2


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

    def __init__(self, num_in=32, num_s=16, mids=4, normalize=False):  # plan 1
        super(FeatureAggregation, self).__init__()
        self.normalize = normalize
        self.num_s = num_s
        self.num_n = mids ** 2
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_proj = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_extend = MyConv(self.num_s, num_in, kernel_size=1, is_bn=False, is_act=False)

        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # self.gcn = GCN1(self.num_s, self.num_n, self.num_s)

    def forward(self, x, edge):
        b, c, h, w = x.size()
        edge = F.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(b, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(b, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(b, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # Project and graph reason
        x_b_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))

        if self.normalize:
            x_b_state = x_b_state * (1. / x_state_reshaped.size(2))

        # adj = torch.randn(self.num_s, self.num_s).cuda()
        # adj = (adj + adj.T) / 2  # 确保邻接矩阵是对称的
        # adj = F.softmax(adj, dim=1)  # 归一化邻接矩阵
        # x_b_rel = self.gcn(x_b_state, adj)
        x_b_rel = self.gcn(x_b_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_b_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(b, self.num_s, *x.size()[2:])
        out = x + self.conv_extend(x_state)

        return out


# def __init__(self, in_channels=64):  # plan 2
#     super(FeatureAggregation, self).__init__()
#     self.query_conv = MyConv(in_channels, in_channels >> 1, kernel_size=1, stride=1, padding=0,
#                              use_bias=True, is_bn=False, is_act=False)
#     self.key_conv = MyConv(in_channels, in_channels >> 1, kernel_size=1, stride=1, padding=0,
#                            use_bias=True, is_bn=False, is_act=False)
#     self.value_conv = MyConv(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
#                              is_bn=False, is_act=False)
#     self.gamma = nn.Parameter(torch.zeros(1))
#     self.softmax = nn.Softmax(dim=1)
#     self.scconv = MyConv(in_channels, in_channels >> 1, kernel_size=1, is_bn=False, is_act=False)
#
# def forward(self, x1, x2):
#     x = torch.cat([x1, x2], dim=1)
#     b, c, h, w = x.size()
#     q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
#     k = self.key_conv(x).view(b, -1, h * w)
#     am = torch.bmm(q, k)
#     a = self.softmax(am)
#     out = torch.bmm(v, a.permute(0, 2, 1))
#     out = out.view(b, c, h, w)
#     out = self.gamma * out + x
#     out = self.scconv(out)
#     return out


# class FeatureAggregation(nn.Module):  # plan 3
#
#     def __init__(self, input_dim=32):
#         super().__init__()
#
#         self.conv = MyConv(input_dim << 1, input_dim, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, h_feature, l_feature):  # ba_res, sa_res
#         w_h_feature = self.sigmoid(h_feature)
#         w_l_feature = self.sigmoid(l_feature)
#
#         l_feature = l_feature + l_feature * w_l_feature + (1 - w_l_feature) * (w_h_feature * h_feature)
#         h_feature = h_feature + h_feature * w_h_feature + (1 - w_h_feature) * (w_l_feature * l_feature)
#
#         out = self.conv(torch.cat([h_feature, l_feature], dim=1))
#         return out

# TODO: GLSA 充当融合模块
# class FeatureAggregation(nn.Module):
#
#     def __init__(self, in_channels=32):
#         super().__init__()
#
#         self.local_conv = nn.Conv2d(in_channels, in_channels, 1)
#         self.global_conv = nn.Conv2d(in_channels, in_channels, 1)
#         self.global_block = ContextBlock(in_planes=in_channels, ratio=2)
#         self.local_branch = ConvBranch(in_features=in_channels)
#         self.conv_restore = MyConv(in_channels * 2, in_channels, 1)
#
#     def forward(self, x0, x1):  # x0: high level => ba_res, x1: low level => sa_res
#         # local block
#         loc = self.local_branch(self.local_conv(x0))
#         # global block
#         glo = self.global_block(self.global_conv(x1))
#         # concat global & local feature
#         cat = torch.cat([loc, glo], dim=1)
#         # restore x's channel to 32
#         fa_res = self.conv_restore(cat)
#         return fa_res
#
#
# class ContextBlock(nn.Module):
#
#     def __init__(self, in_planes, ratio, pooling_type='att', fusion_types=('channel_mul',)):
#         super(ContextBlock, self).__init__()
#
#         assert pooling_type in ['avg', 'att']
#         assert isinstance(fusion_types, (list, tuple))
#         valid_fusion_types = ['channel_add', 'channel_mul']
#         assert all([f in valid_fusion_types for f in fusion_types])
#         assert len(fusion_types) > 0, 'at least one fusion should be used'
#
#         self.in_planes = in_planes
#         self.out_planes = int(in_planes * ratio)
#         self.pooling_type = pooling_type
#
#         if pooling_type == 'att':
#             self.conv_mask = nn.Conv2d(in_planes, 1, kernel_size=1)
#             self.softmax = nn.Softmax(dim=2)
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         if 'channel_add' in fusion_types:
#             self.channel_add_conv = nn.Sequential(
#                 nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1),
#                 nn.LayerNorm([self.out_planes, 1, 1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.out_planes, self.in_planes, kernel_size=1))
#         else:
#             self.channel_add_conv = None
#
#         if 'channel_mul' in fusion_types:
#             self.channel_mul_conv = nn.Sequential(
#                 nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1),
#                 nn.LayerNorm([self.out_planes, 1, 1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.out_planes, self.in_planes, kernel_size=1))
#         else:
#             self.channel_mul_conv = None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.pooling_type == 'att':
#             nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
#             self.conv_mask.inited = True
#
#         if self.channel_add_conv is not None:
#             nn.init.zeros_(self.channel_add_conv[-1].bias)
#             nn.init.zeros_(self.channel_add_conv[-1].weight)
#
#         if self.channel_mul_conv is not None:
#             nn.init.zeros_(self.channel_mul_conv[-1].bias)
#             nn.init.zeros_(self.channel_mul_conv[-1].weight)
#
#     def spatial_pool(self, x):
#         b, c, h, w = x.size()
#
#         if self.pooling_type == 'att':
#             input_x = x
#             input_x = input_x.view(b, c, h * w)
#             input_x = input_x.unsqueeze(1)
#
#             context_mask = self.conv_mask(x)
#             context_mask = context_mask.view(b, 1, h * w)
#
#             context_mask = self.softmax(context_mask)
#             context_mask = context_mask.unsqueeze(-1)
#             context = torch.matmul(input_x, context_mask)
#             context = context.view(b, c, 1, 1)
#         else:
#             context = self.avg_pool(x)
#
#         return context
#
#     def forward(self, x):
#         context = self.spatial_pool(x)
#         out = x
#
#         if self.channel_mul_conv is not None:
#             channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
#             out = out + out * channel_mul_term
#
#         if self.channel_add_conv is not None:
#             channel_add_term = self.channel_add_conv(context)
#             out = out + channel_add_term
#
#         return out
#
#
# class ConvBranch(nn.Module):
#
#     def __init__(self, in_features, hidden_features=None, out_features=None):
#         super().__init__()
#
#         hidden_features = hidden_features or in_features
#         out_features = out_features or in_features
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, 1, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.ReLU(inplace=True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.ReLU(inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.ReLU(inplace=True)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.SiLU(inplace=True)
#         )
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
#             nn.BatchNorm2d(hidden_features),
#             nn.ReLU(inplace=True)
#         )
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(hidden_features, out_features, 1, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.sigmoid_spatial = nn.Sigmoid()
#
#     def forward(self, x):
#         res1 = x
#         res2 = x
#         x = self.conv1(x)
#         x = x + self.conv2(x)
#         x = self.conv3(x)
#         x = x + self.conv4(x)
#         x = self.conv5(x)
#         x = x + self.conv6(x)
#         x = self.conv7(x)
#         x_mask = self.sigmoid_spatial(x)
#         res1 = res1 * x_mask
#         return res2 + res1
