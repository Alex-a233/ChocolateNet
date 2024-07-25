import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.hardnet_68 import HarDNet


class Supervisor(nn.Module):  # refers from MSNet

    def __init__(self):
        super(Supervisor, self).__init__()
        hardnet = HarDNet(arch=68)
        state_dict = torch.load('./pretrained_args/hardnet68.pth')
        hardnet.load_state_dict(state_dict)
        self.layers = hardnet.base
        for layer in self.layers:
            layer.requires_grad_(False)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, mask):
        _, c, _, _ = pred.shape

        if c != 3:
            pred = pred.repeat(1, 3, 1, 1)
            mask = mask.repeat(1, 3, 1, 1)

        pred = (pred - self.mean) / self.std
        mask = (mask - self.mean) / self.std

        sup_loss = 0.0
        for i, layer in enumerate(self.layers):
            pred = layer(pred)
            mask = layer(mask)
            if i in [9, 12, 15]:
                sup_loss += iou_loss(pred, mask)

        return sup_loss / 3.0


def iou_loss(pred, mask):  # TODO: after model converged, try biou
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


if __name__ == '__main__':
    img = cv2.imread('D://Study/pyspace/PraNet/results/PraNet/ETIS-LaribPolypDB/29.png',
                     cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = torch.from_numpy(img).reshape(1, 1, img.shape[0], img.shape[1])
    img = F.interpolate(img, mode='bilinear', size=(352, 352), align_corners=True)

    gt = cv2.imread('D://Study/pyspace/PraNet/data/TestDataset/ETIS-LaribPolypDB/masks/29.png',
                    cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    gt = torch.from_numpy(gt).reshape(1, 1, gt.shape[0], gt.shape[1])
    gt = F.interpolate(gt, mode='bilinear', size=(352, 352), align_corners=True)
    sup = Supervisor()
    loss = sup(img, gt)
    print(loss)
