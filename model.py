import torch
import torch.nn as nn

# from backbone.pvtv2 import PvtV2B2
from backbone.pvtv2 import pvt_v2_b2
from utils.attn import BoundaryAttention, StructureAttention
from utils.toy_block import FeatureAggregation


class ChocolateNet(nn.Module):

    def __init__(self):
        super(ChocolateNet, self).__init__()
        # backbone pvt_v2_b2
        self.backbone = pvt_v2_b2()
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

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # (bs, 64, 88, 88)
        x2 = pvt[1]  # (bs, 128, 44, 44)
        x3 = pvt[2]  # (bs, 320, 22, 22)
        x4 = pvt[3]  # (bs, 512, 11, 11)
        ba_res = self.ba(x2, x3, x4)  # [(bs, 1, 11, 11), (bs, 1, 22, 22), (bs, 1, 44, 44)]
        sa_res = self.sa(x1)  # (bs, 64, 88, 88)
        pred = self.fa(ba_res, sa_res)
        return pred


if __name__ == '__main__':
    model = ChocolateNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    pred = model(x)
    print(pred.shape)
