import cv2
import torch
import torch.nn as nn
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


def bar_ce_loss(pred, target):
    pred_boundary = generate_block_target(pred, 2) == 1
    target_boundary = generate_block_target(target, 2) == 1
    boundary_region = pred_boundary | target_boundary
    boundary_region = (boundary_region >= 0.5)
    loss_mask = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    loss = loss_mask[boundary_region].sum() / boundary_region.sum().clamp(min=1).float()

    w = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    pred = torch.sigmoid(pred)
    eps = 1e-8
    inter = ((pred * target) * w).sum(dim=(2, 3))
    union = ((pred + target) * w).sum(dim=(2, 3))
    w_dice = 1 - (2 * inter + eps) / (union + eps)

    # TODO: generate real mask pred, set boundary width as 1, same as inference !!! most important step
    pred_boundary = generate_block_target(pred, 1) == 1
    pred_boundary = pred_boundary >= 0.5
    pred = pred.sigmoid() >= 0.5

    return loss, w_dice


def generate_block_target(mask_target, boundary_width=3):  # 文中的近似计算方法, 实际上boundary_width=2
    mask_target = mask_target.float()  # 实例掩码 M^k

    # boundary region ~ 边界区域
    kernel_size = 2 * boundary_width + 1  # 5 (当宽度为 1 时，这个核大小变成 3，即文中所谓的 3x3 算子)
    # 要计算宽度为 2 的边界区域，算子可以定义为 5x5 的矩阵
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    # 将算子的中心点设置为核大小的平方-1(bw=2, ks=5 => 5^2-1=24|bw=1, ks=3 => 3^2-1=8)
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

    # 给实例掩码 mask_target 增加一个维度，然后在其上下左右各加入两层 0 填充作为填充之后的目标掩码
    pad_target = F.pad(mask_target, (boundary_width, boundary_width, boundary_width, boundary_width))

    # pos_boundary~前景边界区域
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets

    # neg_boundary~背景边界区域
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)  # 反转 M^k 的二进制值
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets

    # generate block target~B^k
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2

    return block_target


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

    loss, dice = bar_ce_loss(pred, mask)
    print('bar_ce+wdice = ', (loss + dice).item())