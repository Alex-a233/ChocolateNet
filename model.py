import torch
import torch.nn as nn

from backbone.pvtv2 import PvtV2B2
from utils.attn import BoundaryAttention, StructureAttention
from utils.toy_block import FeatureAggregation, MyConv


class ChocolateNet(nn.Module):

    def __init__(self):
        super(ChocolateNet, self).__init__()
        self.backbone = PvtV2B2()
        state_dict = torch.load('./pretrained_args/pvt_v2_b2.pth')
        model_state_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        self.backbone.load_state_dict(model_state_dict)

        # 1. Boundary Attention Module
        self.ba = BoundaryAttention()
        # 2. Structure Attention Module(CA+SA)
        self.sa = StructureAttention()
        # 3. Feature Aggregation Module
        self.fa = FeatureAggregation()

        self.out_ba = MyConv(32, 1, 1, use_bias=True, is_bn=False, is_act=False)
        self.out_fa = MyConv(32, 1, 1, use_bias=True, is_bn=False, is_act=False)

        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # (bs, 64, 88, 88)
        x2 = pvt[1]  # (bs, 128, 44, 44)
        x3 = pvt[2]  # (bs, 320, 22, 22)
        x4 = pvt[3]  # (bs, 512, 11, 11)
        ba_res = self.ba(x2, x3, x4)  # (bs, 32, 44, 44)
        sa_res = self.sa(x1)  # (bs, 32, 44, 44)
        fa_res = self.fa(ba_res, sa_res)  # (bs, 32, 44, 44)
        ba_res = self.out_ba(ba_res)  # (bs, 1, 44, 44)
        fa_res = self.out_fa(fa_res)  # (bs, 1, 44, 44)
        pred1 = self.up(ba_res)
        pred2 = self.up(fa_res)
        return pred1 + pred2


if __name__ == '__main__':
    model = ChocolateNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    import time

    s = time.time()
    pred = model(x)
    e = time.time()
    print(e - s)
    print(pred.shape)
