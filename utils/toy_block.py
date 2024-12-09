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

        # self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn = GCN1(self.num_s, self.num_n, self.num_s)

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

        adj = torch.randn(self.num_s, self.num_s).cuda()
        adj = (adj + adj.T) / 2  # 确保邻接矩阵是对称的
        adj = F.softmax(adj, dim=1)  # 归一化邻接矩阵
        x_b_rel = self.gcn(x_b_state, adj)
        # x_b_rel = self.gcn(x_b_state)

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
#     v = self.value_conv(x).view(b, -1, h * w)
#     out = torch.bmm(v, a.permute(0, 2, 1))
#     out = out.view(b, c, h, w)
#     out = self.gamma * out + x
#     out = self.scconv(out)
#     return out
