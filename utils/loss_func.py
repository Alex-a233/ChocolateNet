import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def wbce_wiou(pred, mask):
    """
    结构损失，即模型的损失函数
    pred: 预测图
    mask: 基准图
    """
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
    eps = 1e-8
    inter = ((pred * mask) * w).sum(dim=(2, 3))
    union = ((pred + mask) * w).sum(dim=(2, 3))
    w_dice = 1 - (2 * inter + eps) / (union + eps)

    return w_bce, w_dice


def bce_dice(pred, mask):
    ce_loss = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter / (union + 1)).mean()
    return ce_loss, dice_loss


class BARCrossEntropyLoss(nn.Module):  # after optimize
    # 实际上 stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0]
    def __init__(self, stage_instance_loss_weight=[1.0, 1.0, 1.0, 1.0], boundary_width=2, start_stage=1):
        super(BARCrossEntropyLoss, self).__init__()
        self.stage_instance_loss_weight = stage_instance_loss_weight
        self.boundary_width = boundary_width
        self.start_stage = start_stage

    def forward(self, stage_instance_preds, stage_instance_targets):
        pre_pred = 0.0
        loss_mask_set = []
        for idx in range(len(stage_instance_preds)):  # 遍历预测图取下标
            # 取当前下标指向的预测图和准确标记
            instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[idx]
            if idx <= self.start_stage:  # 若下标小于等于起始阶段数
                # 计算当前下标指向的预测图和准确标记之间的二元交叉熵损失
                loss_mask = F.binary_cross_entropy(instance_pred, instance_target)
                loss_mask_set.append(loss_mask)  # 将当前损失加入到损失掩码列表中
                pre_pred = instance_pred.sigmoid() >= 0.5  # 取实例预测图中大于等于0.5的部分赋值给 pre_pred

            else:  # 若小标大于起始阶段数
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=self.boundary_width) == 1
                boundary_region = pre_boundary.unsqueeze(1)

                target_boundary = generate_block_target(stage_instance_targets[idx - 1].float(),
                                                        boundary_width=self.boundary_width) == 1
                boundary_region = boundary_region | target_boundary.unsqueeze(1)

                boundary_region = F.interpolate(boundary_region.float(), instance_pred.shape[-2:], mode='bilinear',
                                                align_corners=True)
                boundary_region = (boundary_region >= 0.5).squeeze(1)

                loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target, reduction='none')
                loss_mask = loss_mask[boundary_region].sum() / boundary_region.sum().clamp(min=1).float()
                loss_mask_set.append(loss_mask)

                # generate real mask pred, set boundary width as 1, same as inference
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=1) == 1

                pre_boundary = F.interpolate(pre_boundary.unsqueeze(1).float(), instance_pred.shape[-2:],
                                             mode='bilinear', align_corners=True) >= 0.5

                pre_pred = F.interpolate(stage_instance_preds[idx - 1], instance_pred.shape[-2:], mode='bilinear',
                                         align_corners=True)

                pre_pred[pre_boundary] = stage_instance_preds[idx][pre_boundary]
                pre_pred = pre_pred.squeeze(1).sigmoid() >= 0.5

        assert len(self.stage_instance_loss_weight) == len(loss_mask_set)  # loss_mask_set 的长度是 4
        loss_instance = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)])

        return loss_instance


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
    pad_target = F.pad(mask_target.unsqueeze(1), (boundary_width, boundary_width, boundary_width, boundary_width))

    # pos_boundary~前景边界区域
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets.squeeze(1)

    # neg_boundary~背景边界区域
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)  # 反转 M^k 的二进制值
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets.squeeze(1)

    # generate block target~B^k
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2

    return block_target


if __name__ == '__main__':
    img = cv2.imread('D://Study/pyspace/PraNet/results/PraNet/ETIS-LaribPolypDB/125.png', cv2.IMREAD_GRAYSCALE) / 1.
    img = torch.from_numpy(img).reshape(1, 1, img.shape[0], img.shape[1])
    img = F.interpolate(img, mode='bilinear', size=(352, 352), align_corners=True)

    gt = cv2.imread('D://Study/pyspace/PraNet/data/TestDataset/ETIS-LaribPolypDB/masks/125.png',
                    cv2.IMREAD_GRAYSCALE) / 255.
    gt = torch.from_numpy(gt).reshape(1, 1, gt.shape[0], gt.shape[1])
    gt = F.interpolate(gt, mode='bilinear', size=(352, 352), align_corners=True)

    wbce, wdice = wbce_wdice(img, gt)
    print('wbce = ', wbce.item())
    print('wdice = ', wdice.item())
    print('wbce+wdice = ', (wbce + wdice).mean().item())
    print('wbce+wiou = ', wbce_wiou(img, gt).item())
