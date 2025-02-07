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

    def __init__(self, num_in=32, num_s=16, mids=4):  # plan 1
        super(FeatureAggregation, self).__init__()
        self.num_s = num_s
        self.num_n = mids ** 2
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_proj = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_extend = MyConv(self.num_s, num_in, kernel_size=1, is_bn=False, is_act=False)

        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

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
        x_b_rel = self.gcn(x_b_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_b_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(b, self.num_s, *x.size()[2:])
        out = x + self.conv_extend(x_state)
        return out

    # def forward(self, x1, x2):
    #     b, c, h, w = x1.size()
    #     t1 = x1
    #     # Construct projection matrix
    #     x_state_reshaped = self.conv_state(x1).view(b, self.num_s, -1)
    #     x_proj = self.conv_proj(x1)
    #     x2 = F.softmax(x2, dim=1)[:, 1, :, :].unsqueeze(1)
    #     x_mask = x_proj * x2
    #     x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(b, self.num_s, -1)
    #     x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(b, self.num_s, -1))
    #     x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
    #     x_rproj_reshaped = x_proj_reshaped
    #
    #     # Project and graph reason
    #     x_b_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
    #     x_b_rel = self.gcn(x_b_state)
    #
    #     # Reproject
    #     x_state_reshaped = torch.matmul(x_b_rel, x_rproj_reshaped)
    #     x_state = x_state_reshaped.view(b, self.num_s, h, w)
    #     out = t1 + self.conv_extend(x_state)
    #     return out