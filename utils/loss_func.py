import cv2
import torch
import torch.nn.functional as F


def wbce_wiou(preds, masks):
    """
    结构损失，即模型的损失函数
    preds: 预测图
    masks: 基准图
    """
    w = 1 + 5 * torch.abs(F.avg_pool2d(masks, kernel_size=31, stride=1, padding=15) - masks)
    w_bce = F.binary_cross_entropy_with_logits(preds, masks, reduce=None)
    w_bce = (w * w_bce).sum(dim=(2, 3)) / w.sum(dim=(2, 3))

    preds = torch.sigmoid(preds)
    inter = ((preds * masks) * w).sum(dim=(2, 3))
    union = ((preds + masks) * w).sum(dim=(2, 3))
    w_iou = 1 - (inter + 1) / (union - inter + 1)

    return torch.mean(w_bce + w_iou)


# dice 偏于医学图像，故而借鉴尝试
def wbce_wdice(preds, masks):
    w = 1 + 5 * torch.abs(F.avg_pool2d(masks, kernel_size=31, stride=1, padding=15) - masks)
    w_bce = F.binary_cross_entropy_with_logits(preds, masks, reduce=None)
    w_bce = (w * w_bce).sum(dim=(2, 3)) / w.sum(dim=(2, 3))

    preds = torch.sigmoid(preds)
    eps = 1e-8
    inter = ((preds * masks) * w).sum(dim=(2, 3))
    union = ((preds + masks) * w).sum(dim=(2, 3))
    w_dice = 1 - (2 * inter + eps) / (union + eps)

    return w_bce, w_dice


def bce_dice(preds, masks):
    ce_loss = F.binary_cross_entropy_with_logits(preds, masks)
    preds = torch.sigmoid(preds)
    inter = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter / (union + 1)).mean()
    return ce_loss, dice_loss


if __name__ == '__main__':
    pred = cv2.imread('D://Study/pyspace/PraNet/results/PraNet/ETIS-LaribPolypDB/125.png', cv2.IMREAD_GRAYSCALE) / 1.
    pred = torch.from_numpy(pred).reshape(1, 1, pred.shape[0], pred.shape[1])
    pred = F.interpolate(pred, mode='bilinear', size=(352, 352), align_corners=True)

    mask = cv2.imread('D://Study/pyspace/PraNet/data/TestDataset/ETIS-LaribPolypDB/masks/125.png',
                      cv2.IMREAD_GRAYSCALE) / 255.
    mask = torch.from_numpy(mask).reshape(1, 1, mask.shape[0], mask.shape[1])
    mask = F.interpolate(mask, mode='bilinear', size=(352, 352), align_corners=True)

    wbce, wdice = wbce_wdice(pred, mask)
    print('wbce = ', wbce.item())
    print('wdice = ', wdice.item())
    print('wbce+wdice = ', (wbce + wdice).mean().item())
    print('wbce+wiou = ', wbce_wiou(pred, mask).item())
