import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.hardnet_68 import HarDNet


class Supervisor(nn.Module):  # refers from MSNet

    # def __init__(self):
    #     super(Supervisor, self).__init__()
    #     res2net = Res2Net50()
    #
    #     layer0 = nn.Sequential(res2net.conv1, res2net.bn1, nn.ReLU(inplace=True))
    #     layer1 = res2net.layer1
    #     layer2 = res2net.layer2
    #     layer3 = res2net.layer3
    #     layer4 = res2net.layer4
    #
    #     for layer in [layer0, layer1, layer2, layer3, layer4]:
    #         for m in layer:
    #             m.requires_grad = False
    #
    #     self.layers = nn.ModuleList([layer0, layer1, layer2, layer3, layer4])
    #     self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    #     self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    #
    # def forward(self, pred, mask):
    #     _, c, h, _ = pred.shape
    #
    #     if c != 3:
    #         pred = pred.repeat(1, 3, 1, 1)
    #         mask = mask.repeat(1, 3, 1, 1)
    #
    #     pred = (pred - self.mean) / self.std
    #     mask = (mask - self.mean) / self.std
    #
    #     mse_loss = 0.0
    #     for layer in self.layers:
    #         pred = layer(pred)
    #         mask = layer(mask)
    #         mse_loss = mse_loss + F.mse_loss(pred, mask)
    #
    #     return mse_loss
    def __init__(self):
        super(Supervisor, self).__init__()
        # TODO: this could be replaced by res2net50, change it maybe increasing the training speed
        hardnet = HarDNet(arch=68)
        state_dict = torch.load('./pretrained_args/hardnet68.pth')
        hardnet.load_state_dict(state_dict)
        self.layers = hardnet.base
        for layer in self.layers:
            layer.requires_grad_(False)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, mask):
        _, c, h, _ = pred.shape

        if c != 3:
            pred = pred.repeat(1, 3, 1, 1)
            mask = mask.repeat(1, 3, 1, 1)

        pred = (pred - self.mean) / self.std
        mask = (mask - self.mean) / self.std

        if h != 224:
            pred = F.interpolate(pred, mode='bilinear', size=(224, 224), align_corners=False)
            mask = F.interpolate(mask, mode='bilinear', size=(224, 224), align_corners=False)

        mse_loss = 0.0
        for i, layer in enumerate(self.layers):  # 17 - 3
            pred = layer(pred)
            mask = layer(mask)
            if i in [9, 12, 15]:
                mse_loss += F.mse_loss(pred, mask)

        return mse_loss


if __name__ == '__main__':
    img = cv2.imread('D://Study/pyspace/PraNet/results/PraNet/ETIS-LaribPolypDB/29.png',
                     cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = torch.from_numpy(img).resize(1, 1, img.shape[0], img.shape[1])
    img = torch.nn.functional.interpolate(img, mode='bilinear', size=(352, 352), align_corners=True)

    gt = cv2.imread('D://Study/pyspace/PraNet/data/TestDataset/ETIS-LaribPolypDB/masks/29.png',
                    cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    gt = torch.from_numpy(gt).resize(1, 1, gt.shape[0], gt.shape[1])
    gt = torch.nn.functional.interpolate(gt, mode='bilinear', size=(352, 352), align_corners=True)
    sup = Supervisor()
    loss = sup(img, gt)
    print(loss)
