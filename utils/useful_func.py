import os

import numpy as np
import torch.nn.functional as F

from utils.dataloader import TestSet


# 打印和保存日志信息
def print_save(info, path, file_name):
    print(info)
    empty_create(path)
    f = open(path + file_name, 'a')
    f.write(info + '\n')
    f.close()


# 不存在则创建路径
def empty_create(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 梯度剪裁 TODO:训练较为稳定再加入这个
def clip_gradient(optimizer, grad_clip):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# 计算时间损耗
def calculate_time_loss(start_time, end_time, process_type):
    cost_time = str(end_time - start_time).split(':')
    hours = int(cost_time[0])
    minutes = int(cost_time[1])
    seconds = int(cost_time[2].split('.')[0])
    print('%s took %s hours %s minutes %s seconds' % (process_type, hours, minutes, seconds))


# 选择最佳模型权重参数
def choose_best(model, args):
    eval_path = args.eval_path
    record = {}
    image_num = 0
    dice_sum = 0.0
    eps = 1e-8

    for dataset in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:  # 'BKAI-IGH-NEOPOLYP',
        image_path = os.path.join(eval_path, dataset, 'images/')
        mask_path = os.path.join(eval_path, dataset, 'masks/')
        image_list = [f for f in os.listdir(image_path) if f.endswith('.png')]
        cur_image_num = len(image_list)
        cur_dice_sum = 0.
        image_num += cur_image_num

        testset_loader = TestSet(image_path, mask_path, args.eval_size)

        for i in range(cur_image_num):
            image, mask, name = testset_loader.load_data()

            mask = np.asarray(mask, np.float32)
            mask /= (mask.max(initial=.0) + eps)

            image = image.cuda()
            pred = model(image)

            pred = F.interpolate(pred, size=mask.shape, mode='bilinear', align_corners=False)
            pred = pred.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)

            pred = np.reshape(pred, -1)
            mask = np.reshape(mask, -1)

            intersection = pred * mask
            dice = (2 * intersection.sum() + eps) / (pred.sum() + mask.sum() + eps)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            cur_dice_sum += dice

        cur_mdice = cur_dice_sum / cur_image_num
        record[dataset] = cur_mdice
        dice_sum += cur_dice_sum

    return dice_sum / image_num, record


if __name__ == '__main__':
    'test function `print_save'
    # print_save('hhhh', './save_dir/', 'test1.txt')

    'test function `choose_best`'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_path', default='../dataset/testset')
    # parser.add_argument('--eval_size', default=352)
    # args = parser.parse_args()
    # model = ChocolateNet().cuda()
    # mdice, record = choose_best(model, args)
    # print(mdice)

    'test function `calculate_time_loss`'
    # start = datetime.datetime.now()
    # time.sleep(100)
    # end = datetime.datetime.now()
    # calculate_time_loss(start, end, 'Unit Test')
