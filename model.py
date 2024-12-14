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
        model_state_dict_keys = model_state_dict.keys()
        state_dict_items = state_dict.items()
        state_dict = {k: v for k, v in state_dict_items if k in model_state_dict_keys}
        model_state_dict.update(state_dict)
        self.backbone.load_state_dict(model_state_dict)

        # 1. Boundary Attention Module(RA+SA)
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
        # 换成仅 pred2 性能会下降
        # return pred1 * pred2  # 性能尚可
        return pred1 + pred2


if __name__ == '__main__':
    model = ChocolateNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    pred = model(x)
    print(pred.shape)

    # Polyp-PVT's performance
    # CVC-300                       mdice: 0.880  miou: 0.802
    # CVC-ClinicDB             mdice: 0.937   miou: 0.889
    # CVC-ColonDB             mdice: 0.808  miou: 0.727
    # ETIS-LaribPolypDB mdice: 0.787   miou: 0.706
    # Kvasir                            mdice: 0.917    miou: 0.864
