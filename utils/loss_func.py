import cv2
import torch
import torch.nn.functional as F

'''
结构损失，即模型的损失函数
pred: 预测图
mask: 基准图
'''


def wbce_wiou(pred, mask):
    w = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    w_bce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    w_bce = (w * w_bce).sum(dim=(2, 3)) / w.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * w).sum(dim=(2, 3))
    union = ((pred + mask) * w).sum(dim=(2, 3))
    w_iou = 1 - (inter + 1) / (union - inter + 1)

    return (w_bce + w_iou).mean()


# dice 偏于医学图像，故而借鉴尝试
def wbce_wdice(pred, mask):
    w = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    w_bce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    w_bce = (w * w_bce).sum(dim=(2, 3)) / w.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * w).sum(dim=(2, 3))
    union = ((pred + mask) * w).sum(dim=(2, 3))
    eps = 1e-8
    w_dice = 1 - ((2 * inter + eps) / (union + eps))

    return w_bce, w_dice


def bce_dice(pred, mask):
    ce_loss = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter / (union + 1)).mean()
    return ce_loss, dice_loss


if __name__ == '__main__':
    img = cv2.imread('D://Study/pyspace/PraNet/results/PraNet/ETIS-LaribPolypDB/2.png', cv2.IMREAD_GRAYSCALE) / 255.0
    img = torch.from_numpy(img).reshape(1, 1, img.shape[0], img.shape[1])
    img = torch.nn.functional.interpolate(img, mode='bilinear', size=(352, 352), align_corners=True)

    gt = cv2.imread('D://Study/pyspace/PraNet/data/TestDataset/ETIS-LaribPolypDB/masks/2.png', cv2.IMREAD_GRAYSCALE) / 255.0
    gt = torch.from_numpy(gt).reshape(1, 1, gt.shape[0], gt.shape[1])
    gt = torch.nn.functional.interpolate(gt, mode='bilinear', size=(352, 352), align_corners=True)

    wbce, wdice = wbce_wdice(gt, gt)
    print('wbce = ', wbce.item())
    print('wdice = ', wdice.item())
    print('wbce+wdice = ', (wbce + wdice).mean().item())
    print('wbce+wiou = ', wbce_wiou(gt, gt).item())
