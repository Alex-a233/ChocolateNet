import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ptflops import get_model_complexity_info
from skimage.measure import regionprops, label, find_contours
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from backbone.pvtv2 import PvtV2B2
from model import ChocolateNet
from study.model1 import ChocolateNet1
from utils.dataloader import TrainSet, TestSet
from utils.toy_block import MyConv, GCN


def progress_bar():
    num_mask = 100
    for _ in tqdm(range(num_mask), colour='#e946ef'):
        time.sleep(1 / 10)


def experiment_of_dye():
    all_color_maps = (
        'COLORMAP_AUTUMN',  # 0
        'COLORMAP_BONE',  # 1
        'COLORMAP_JET',  # 2
        'COLORMAP_WINTER',  # 3
        'COLORMAP_RAINBOW',  # 4
        'COLORMAP_OCEAN',  # 5
        'COLORMAP_SUMMER',  # 6
        'COLORMAP_SPRING',  # 7
        'COLORMAP_COOL',  # 8
        'COLORMAP_HSV',  # 9
        'COLORMAP_PINK',  # 10
        'COLORMAP_HOT',  # 11
        'COLORMAP_PARULA',  # 12
        'COLORMAP_MAGMA',  # 13
        'COLORMAP_INFERNO',  # 14
        'COLORMAP_PLASMA',  # 15
        'COLORMAP_VIRIDIS',  # 16
        'COLORMAP_CIVIDIS',  # 17
        'COLORMAP_TWILIGHT',  # 18
        'COLORMAP_TWILIGHT_SHIFTED',  # 19
        'COLORMAP_TURBO',  # 20
        'COLORMAP_DEEPGREEN'  # 21
    )
    chosen_color_maps = {
        'COLORMAP_BONE': 1,
        'COLORMAP_OCEAN': 5,
        'COLORMAP_PINK': 10,
        'COLORMAP_VIRIDIS': 16,
        'COLORMAP_CIVIDIS': 17,
        'COLORMAP_DEEPGREEN': 21
    }
    base_path = 'D://Study/pyspace/PraNet/data/TrainDataset/images/'
    img_paths = [base_path + img for img in os.listdir(base_path) if img.endswith('.png')]
    for img_path in img_paths:
        cpr_colormap(img_path, chosen_color_maps)


def cpr_colormap(img_path, color_dict):
    h1 = []
    h2 = []
    origin_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h1.append(origin_img)
    h2.append(origin_img)
    # cv2.imshow('origin_image', origin_img)
    img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    i = 0
    for k, v in color_dict.items():
        dst = cv2.applyColorMap(img, v)
        if i < 3:
            h1.append(dst)
        else:
            h2.append(dst)
        i += 1
        # cv2.imshow(k, dst)
    hc1 = cv2.hconcat(h1)
    hc2 = cv2.hconcat(h2)
    full_img = cv2.vconcat([hc1, hc2])
    cv2.imshow('ORIGIN BONE OCEAN PINK VIRIDIS CIVIDIS DEEPGREEN', full_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_bkai_dataset():
    bkai_path = '../dataset/testset/BKAI-IGH-NEOPOLYP/masks/'
    bkai_list = [bkai_path + dir for dir in os.listdir(bkai_path) if dir.endswith('.png')]
    for bkai_img in bkai_list:
        origin = cv2.imread(bkai_img, cv2.IMREAD_GRAYSCALE)
        new = cv2.imread(bkai_img, cv2.IMREAD_GRAYSCALE)
        new[new > 10] = 255
        hs = [origin, new]
        hc = cv2.hconcat(hs)
        cv2.imshow('gray & binary', hc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def calc_mean_std():
    dataset = SegmentationDataset()
    # 假设 dataset 是一个 PyTorch 的数据集
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 计算均值和方差
    mean = torch.zeros(3)
    std = torch.zeros(3)
    # for img, _ in train_loader:
    for name, img in train_loader:
        print('processing {} ...'.format(name))
        for c in range(3):
            mean[c] += img[:, c, :, :].mean()
            std[c] += img[:, c, :, :].std()
    mean.div_(len(train_loader))
    std.div_(len(train_loader))

    print(f'mean: {mean}')
    print(f'std: {std}')


class SegmentationDataset(Dataset):

    # def __init__(self, root_dir='D:/Study/polyp_dataset'):
    def __init__(self, root_dir='D:\\Study\\pyspace\\SANet\\data\\test\\ETIS-LaribPolypDB'):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'image')) if f.endswith('.png')])
        # self.mask_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'masks')) if f.endswith('.png')])
        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'image', self.image_files[idx])
        # mask_path = os.path.join(self.root_dir, 'masks', self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        # mask = self.image_transform(mask)

        # return image, mask
        return self.image_files[idx], image

    def get_image(self, i):
        image_path = os.path.join(self.root_dir, 'image', self.image_files[i])
        image = Image.open(image_path).convert('RGB')
        return image


def mask_to_border(mask):
    """ Convert a mask to border image """
    h, w = mask.shape  # 准确标记的高，宽
    border = np.zeros((h, w))  # 创建一个全零矩阵 border

    contours = find_contours(mask, 128)  # 大于 128 的区域被识别为轮廓
    for contour in contours:  # 遍历轮廓数组
        for c in contour:  # 然后取里面的坐标
            x = int(c[0])  # x 坐标
            y = int(c[1])  # y 坐标
            border[x][y] = 255  # 将坐标对应的点位赋值为 255

    return border  # 返回轮廓


def mask_to_bbox(mask):
    """ Mask to bounding boxes """
    mask = mask_to_border(mask)  # 先生成准确标记对应的轮廓矩阵
    lbl = label(mask)  # 生成轮廓的连通区域标记

    bboxes = []  # 边界框列表，一个图像里可能有多个息肉

    props = regionprops(lbl)  # 返回标记区域的属性
    for prop in props:  # 遍历标记区域属性 (min_row, min_col, max_row, max_col)
        x1 = prop.bbox[1]  # 取属性的边界框属性里的左上角横坐标
        y1 = prop.bbox[0]  # 左上角纵坐标
        x2 = prop.bbox[3]  # 右下角横坐标
        y2 = prop.bbox[2]  # 右下角纵坐标
        bboxes.append([x1, y1, x2, y2])  # 将这些坐标点存入列表再存入bbox列表

    return bboxes  # 返回边界框


def mask_to_text(mask):  # 将准确标记转为文本
    bboxes = mask_to_bbox(mask)  # 将准确标记转为边界框
    polyp_size = None  # 息肉尺寸

    for bbox in bboxes:  # 遍历边界框
        x1, y1, x2, y2 = bbox  # 取边界框的各个坐标最小列，最小行，最大列，最大行
        h = (y2 - y1)  # 高度
        w = (x2 - x1)  # 宽度
        area = (h * w) / (mask.shape[0] * mask.shape[1])  # 当前息肉占准确标记图像的比例

        if area < 0.10:  # 若比例小于 1/10
            polyp_size = 0  # 息肉的尺寸是 0 => small
        elif 0.10 <= area < 0.30:  # 比例介于 [1/10, 3/10)
            polyp_size = 1  # 息肉的尺寸是 1 => medium
        else:  # 比例大于等于 3/10
            polyp_size = 2  # 息肉的尺寸是 2 => large

    return polyp_size


def calc_polyp_area(mask):
    mask[mask > 128] = True
    area = np.sum(mask)
    h, w = mask.shape
    prop = area / (h * w)

    # 如果肠息肉直径在0.5cm以下属于细小息肉，直径在0.5-1cm之间属于小息肉，直径在1-2cm之间属于中等大小息肉，直径大于2cm属于大息肉。
    if prop <= 0.1:  # 若比例小于等于 1/10
        polyp_size = 0  # 息肉的尺寸是 0 => tiny
    elif prop <= 0.2:  # 若比例小于等于 1/5
        polyp_size = 1  # 息肉的尺寸是 1 => small
    elif prop <= 0.4:  # 比例介于 [1/5, 2/5)
        polyp_size = 2  # 息肉的尺寸是 2 => medium
    else:  # 比例大于 2/5
        polyp_size = 3  # 息肉的尺寸是 3 => large

    return polyp_size


def count_polyp():
    sum_dict = {}  # dataset_name: [total, tiny, small, medium, large]
    polyp_sizes = {0: 0, 1: 0, 2: 0, 3: 0}  # 0: tiny, 1: small, 2: medium, 3: large
    polyp_num = 0

    for testset_name in ['BKAI-IGH-NEOPOLYP', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        data_path = './dataset/testset/{}'.format(testset_name)
        train_path = './dataset/trainset/masks'
        mask_path = '{}/masks/'.format(data_path)
        names = os.listdir(mask_path)
        mask_num = len(names)
        polyp_num += mask_num
        tiny = 0
        small = 0
        medium = 0
        large = 0

        for name in names:
            mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)

            if testset_name == 'BKAI-IGH-NEOPOLYP':
                mask[mask > 10] = 255
            mask = cv2.resize(mask, (352, 352))
            polyp_size = calc_polyp_area(mask)

            tiny += 1 if polyp_size == 0 else 0
            small += 1 if polyp_size == 1 else 0
            medium += 1 if polyp_size == 2 else 0
            large += 1 if polyp_size == 3 else 0

        if testset_name == 'CVC-ClinicDB':
            train_names = [train_name for train_name in os.listdir(train_path) if len(train_name.split('.')[0]) < 4]
            mask_num += len(train_names)
            polyp_num += len(train_names)

            for name in train_names:
                mask = cv2.imread(os.path.join(train_path, name), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (352, 352))
                polyp_size = calc_polyp_area(mask)

                tiny += 1 if polyp_size == 0 else 0
                small += 1 if polyp_size == 1 else 0
                medium += 1 if polyp_size == 2 else 0
                large += 1 if polyp_size == 3 else 0

        if testset_name == 'Kvasir':
            train_names = [train_name for train_name in os.listdir(train_path) if len(train_name.split('.')[0]) > 4]
            mask_num += len(train_names)
            polyp_num += len(train_names)

            for name in train_names:
                mask = cv2.imread(os.path.join(train_path, name), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (352, 352))
                polyp_size = calc_polyp_area(mask)

                tiny += 1 if polyp_size == 0 else 0
                small += 1 if polyp_size == 1 else 0
                medium += 1 if polyp_size == 2 else 0
                large += 1 if polyp_size == 3 else 0

        polyp_sizes[0] += tiny
        polyp_sizes[1] += small
        polyp_sizes[2] += medium
        polyp_sizes[3] += large

        sum_dict[testset_name] = [mask_num, tiny, small, medium, large]
        print('polyps number in {} = {} and the distribution of them is tiny = {}, small = {}, medium = {}, large = {}'
              .format(testset_name, mask_num, tiny, small, medium, large))

    sum_dict['all'] = [polyp_num, polyp_sizes[0], polyp_sizes[1], polyp_sizes[2], polyp_sizes[3]]
    summary = 'polyps number in these 6 datasets = {} and the polyp size distribution is tiny = {}, small = {}, ' \
              'medium = {}, large = {} '
    print(summary.format(polyp_num, polyp_sizes[0], polyp_sizes[1], polyp_sizes[2], polyp_sizes[3]))

    return sum_dict


def draw_graph(sum_dict):
    # set horizontal axis as number
    width = 0.1
    sizes = ['0-0.1', '0.1-0.2', '0.2-0.4', '0.4-1']
    x = np.arange(len(sizes))
    bkai_x = x
    cvc_300_x = x + width
    cvc_clinicdb_x = x + 2 * width
    cvc_colondb_x = x + 3 * width
    etis_x = x + 4 * width
    kvasir_x = x + 5 * width
    plt.rcParams['font.family'] = 'Georgia'
    # 6 bar settings
    plt.bar(bkai_x, [num / sum_dict['BKAI-IGH-NEOPOLYP'][0] for num in sum_dict['BKAI-IGH-NEOPOLYP'][1:]],
            width=width, color='red', label='BKAI-IGH-NEOPOLYP')
    plt.bar(cvc_300_x, [num / sum_dict['CVC-300'][0] for num in sum_dict['CVC-300'][1:]],
            width=width, color='orange', label='CVC-300')
    plt.bar(cvc_clinicdb_x, [num / sum_dict['CVC-ClinicDB'][0] for num in sum_dict['CVC-ClinicDB'][1:]],
            width=width, color='yellow', label='CVC-ClinicDB')
    plt.bar(cvc_colondb_x, [num / sum_dict['CVC-ColonDB'][0] for num in sum_dict['CVC-ColonDB'][1:]],
            width=width, color='lime', label='CVC-ColonDB')
    plt.bar(etis_x, [num / sum_dict['ETIS-LaribPolypDB'][0] for num in sum_dict['ETIS-LaribPolypDB'][1:]],
            width=width, color='magenta', label='ETIS-LaribPolypDB')
    plt.bar(kvasir_x, [num / sum_dict['Kvasir'][0] for num in sum_dict['Kvasir'][1:]],
            width=width, color='aqua', label='Kvasir')
    # convert horizontal axis to sizes
    plt.xticks(x + width, labels=sizes)
    plt.xlabel('polyp area prop')
    plt.ylabel('polyp size prop')
    # show graph legend
    plt.legend(loc='best')
    plt.title('polyp size distribution')
    plt.show()


def calc_model_complexity():
    model = ChocolateNet().cuda()
    flops, params = get_model_complexity_info(model, input_res=(3, 352, 352),
                                              as_strings=True, print_per_layer_stat=False)
    print('ChocolateNet\'s flops = {} and its params = {}'.format(flops, params))


def calc_fps(n):
    model = ChocolateNet().cuda()
    start_time = time.perf_counter()

    for i in range(n):
        x = torch.randn(16, 3, 352, 352).cuda()
        model.forward(x)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = n / total_time
    print('ChocolateNet\'s fps = {:2f}'.format(fps))


class BAModule(nn.Module):

    def __init__(self):
        super(BAModule, self).__init__()
        self.backbone = PvtV2B2()
        path = './pretrained_args/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # 最初版本
        # self.conv2 = MyConv(128, 32, 1, is_act=False)
        # self.conv3 = MyConv(320, 32, 1, is_act=False)
        # self.conv4 = MyConv(512, 32, 1, is_act=False)
        #
        # self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        #
        # self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        # self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        #
        # self.conv5 = MyConv(96, 32, 3, padding=1, is_act=False)

        # BA 取后三个特征图，改通道数
        self.conv2 = MyConv(128, 32, 1, use_bias=True)
        self.conv3 = MyConv(320, 32, 1, use_bias=True)
        self.conv4 = MyConv(512, 32, 1, use_bias=True)

        self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.conv5 = MyConv(96, 32, 3, padding=1, use_bias=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 1. 高层次特征图很模糊，不是正常人眼观察范围内的图像，故不可用常规的辨识方式判别 BA 的效果
        pvt = self.backbone(x)
        # x1 = pvt[0]  # (bs, 64, 88, 88)

        x2 = pvt[1]  # (bs, 128, 44, 44)
        x3 = pvt[2]  # (bs, 320, 22, 22)
        x4 = pvt[3]  # (bs, 512, 11, 11)

        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 后两个上采样，三个特征图做减法，减出有差异的边界像素
        x3_2 = self.convs3_2(abs(self.up(x3) - x2))  # 2,3层异同点
        x4_2 = self.convs4_2(abs(self.up(self.up(x4)) - x2))  # 2,4层异同点
        x4_3 = self.convs4_3(abs(self.up(x4) - x3))  # 3,4层异同点
        # x4_3_2 = self.convs4_3_2(x3_2 + x4_2 + self.up(x4_3))  # 2,3,4层异同点
        x4_3_2 = self.convs4_3_2(abs(abs(x3_2 - x4_2) - self.up(x4_3)))

        # origin version
        o3_2 = self.convm3_2(self.up(x3)) * x2 * x3_2
        o4_2 = self.convm4_2(self.up(self.up(x4))) * x2 * x4_2
        o4_3 = self.convm4_3(self.up(x4)) * x3 * x4_3

        res = torch.cat((self.up(o4_3), o4_2, o3_2), dim=1)
        res = self.conv5(res)
        # res = res * x4_3_2
        res = res * x4_3_2 + x2 + self.up(x3) + self.up(self.up(x4))

        res = self.up(res)
        return res


def test_boundary_attention():
    module = BAModule()
    module.cuda()
    parser = argparse.ArgumentParser(description='here is training arguments')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    args = parser.parse_args()

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    conv = nn.Conv2d(32, 3, kernel_size=(1, 1))
    conv.cuda()
    up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    for step, (images, masks) in enumerate(trainset_loader, start=1):
        images = images.cuda().float()
        # masks = masks.cuda().float()
        res = module(images)
        res = up(conv(res))
        res = res.squeeze(0).permute(1, 2, 0).sigmoid().data.cpu().numpy()
        origin = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
        cimg = np.hstack((origin, res))
        cv2.imshow('origin & ba_res', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if step == 10:
            break


class SAModule(nn.Module):

    def __init__(self):
        super(SAModule, self).__init__()
        self.backbone = PvtV2B2()
        path = './pretrained_args/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # SA v1
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc1 = MyConv(64, 4, 1, is_bn=False, is_act=False)
        # self.fc2 = MyConv(4, 64, 1, is_bn=False, is_act=False)
        # self.relu = nn.ReLU()
        #
        # self.conv = MyConv(2, 1, 7, padding=3, is_bn=False, is_act=False)
        # self.conv1 = MyConv(64, 32, 1, use_bias=True, is_act=False)

        # SA v2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = MyConv(64, 64 // 16, 1, is_bn=False, is_act=False)
        self.fc2 = MyConv(64, 64 // 16, 1, is_bn=False, is_act=False)
        self.fc3 = MyConv(64 // 16, 64, 1, is_bn=False, is_act=False)
        self.fc4 = MyConv(64 // 16, 64, 1, is_bn=False, is_act=False)
        self.relu = nn.ReLU()
        self.conv = MyConv(1, 1, 7, padding=3, is_bn=False, is_act=False)
        self.conv1 = MyConv(64, 32, 1, use_bias=True, is_act=False)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # (bs, 64, 88, 88)
        # x2 = pvt[1]  # (bs, 128, 44, 44)
        # x3 = pvt[2]  # (bs, 320, 22, 22)
        # x4 = pvt[3]  # (bs, 512, 11, 11)

        # v1
        # avg_res = self.fc2(self.relu(self.fc1(self.avg_pool(x1))))
        # max_res = self.fc2(self.relu(self.fc1(self.max_pool(x1))))
        # am_res = avg_res + max_res
        # t = torch.sigmoid(am_res) * x1
        #
        # avg_res = torch.mean(t, dim=1, keepdim=True)
        # max_res, _ = torch.max(t, dim=1, keepdim=True)
        # res = torch.cat([avg_res, max_res], dim=1)
        # res = self.conv(res)
        # res = torch.sigmoid(res) * t
        # res0 = self.conv1(res)
        # res = self.conv1(res) + self.conv1(x1)

        # SA v2
        avg_res = self.fc3(self.relu(self.fc1(self.avg_pool(x1))))
        max_res = self.fc4(self.relu(self.fc2(self.max_pool(x1))))
        am_res = avg_res + max_res
        t = torch.sigmoid(am_res) * x1

        avg_res = torch.mean(t, dim=1, keepdim=True)
        max_res, _ = torch.max(t, dim=1, keepdim=True)
        res = avg_res + max_res
        res = self.conv(res)
        res = torch.sigmoid(res) * t
        res0 = self.conv1(res)
        res = self.conv1(res) + self.conv1(x1)

        return res0, res


def test_structure_attention():
    model = SAModule()
    model.cuda()
    parser = argparse.ArgumentParser(description='here is the training arguments')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    args = parser.parse_args()

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    conv = nn.Conv2d(32, 3, kernel_size=(1, 1))
    conv.cuda()
    up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    for step, (images, masks) in enumerate(trainset_loader, start=1):
        images = images.cuda().float()
        # masks = masks.cuda().float()
        res0, res = model(images)
        res0 = up(conv(res0))
        res0 = res0.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

        res = up(conv(res))
        res = res.squeeze(0).permute(1, 2, 0).data.cpu().numpy()

        origin = images.squeeze().permute(1, 2, 0).data.cpu().numpy()
        cimg = np.hstack((origin, res0, res))
        cv2.imshow('origin & sa_res', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if step == 10:
            break


class FAModule(nn.Module):

    def __init__(self, num_in=32, num_s=16, mids=4, normalize=False):
        super(FAModule, self).__init__()
        # backbone
        self.backbone = PvtV2B2()
        path = './pretrained_args/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ba toys
        self.conv2 = MyConv(128, 32, 1, use_bias=True)
        self.conv3 = MyConv(320, 32, 1, use_bias=True)
        self.conv4 = MyConv(512, 32, 1, use_bias=True)

        self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True)
        self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True)

        self.conv5 = MyConv(96, 32, 3, padding=1, use_bias=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # sa toys
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = MyConv(64, 4, 1, is_bn=False, is_act=False)
        self.fc2 = MyConv(4, 64, 1, is_bn=False, is_act=False)
        self.relu = nn.ReLU()

        self.conv = MyConv(2, 1, 7, padding=3, is_bn=False, is_act=False)
        self.conv1 = MyConv(64, 32, 1, use_bias=True, is_act=False)
        # fa toys
        self.normalize = normalize
        self.num_s = num_s
        self.num_n = mids ** 2
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids << 1, mids << 1))

        self.conv_state = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_proj = MyConv(num_in, self.num_s, kernel_size=1, use_bias=True, is_bn=False, is_act=False)
        self.conv_extend = MyConv(self.num_s, num_in, kernel_size=1, is_bn=False, is_act=False)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # (bs, 64, 88, 88)
        x2 = pvt[1]  # (bs, 128, 44, 44)
        x3 = pvt[2]  # (bs, 320, 22, 22)
        x4 = pvt[3]  # (bs, 512, 11, 11)

        # ba forward
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 后两个上采样，三个特征图做减法，减出有差异的边界像素
        x3_2 = self.convs3_2(abs(self.up(x3) - x2))  # 2,3层异同点
        x4_2 = self.convs4_2(abs(self.up(self.up(x4)) - x2))  # 2,4层异同点
        x4_3 = self.convs4_3(abs(self.up(x4) - x3))  # 3,4层异同点
        x4_3_2 = self.convs4_3_2(x3_2 + x4_2 + self.up(x4_3))  # 2,3,4层异同点

        # origin version
        o3_2 = self.convm3_2(self.up(x3)) * x2 * x3_2
        o4_2 = self.convm4_2(self.up(self.up(x4))) * x2 * x4_2
        o4_3 = self.convm4_3(self.up(x4)) * x3 * x4_3

        ba_res = torch.cat((self.up(o4_3), o4_2, o3_2), dim=1)
        ba_res = self.conv5(ba_res)
        ba_res = ba_res * x4_3_2
        ba_res = ba_res * x4_3_2 + x2 + self.up(x3) + self.up(self.up(x4))

        # sa forward
        avg_res = self.fc2(self.relu(self.fc1(self.avg_pool(x1))))
        max_res = self.fc2(self.relu(self.fc1(self.max_pool(x1))))
        am_res = avg_res + max_res
        t = torch.sigmoid(am_res) * x1

        avg_res = torch.mean(t, dim=1, keepdim=True)
        max_res, _ = torch.max(t, dim=1, keepdim=True)
        sa_res = torch.cat([avg_res, max_res], dim=1)
        sa_res = self.conv(sa_res)
        sa_res = torch.sigmoid(sa_res) * t
        sa_res = self.conv1(sa_res)
        sa_res = F.interpolate(sa_res, scale_factor=0.5, mode='bilinear', align_corners=True)

        # fa forward
        b, c, h, w = ba_res.size()
        edge = F.softmax(sa_res, dim=1)[:, 1, :, :].unsqueeze(1)

        # Construct projection matrix
        x_state_reshaped = self.conv_state(ba_res).view(b, self.num_s, -1)
        x_proj = self.conv_proj(ba_res)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 2:-2, 2:-2].reshape(b, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(b, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # Project and graph reason
        x_b_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_b_state = x_b_state * (1. / x_state_reshaped.size(2))
        x_b_rel = self.gcn(x_b_state)
        # x_b_rel = self.cconv(x_b_state)

        # Reproject
        x_state_reshaped = torch.matmul(x_b_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(b, self.num_s, h, w)
        out = ba_res + self.conv_extend(x_state)

        return out


def test_feature_aggregation():
    fa = FAModule()
    fa.cuda()
    parser = argparse.ArgumentParser(description='here is training arguments')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    args = parser.parse_args()

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    scconv = nn.Conv2d(32, 1, kernel_size=(1, 1))
    scconv.cuda()
    up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    for step, (images, masks) in enumerate(trainset_loader, start=1):
        images = images.cuda().float()
        masks = masks.cuda().float()
        res = fa(images)
        res = up(scconv(res))
        res = res.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
        origin = masks.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
        cimg = np.hstack((origin, res))
        cv2.imshow('fa_res', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if step == 10:
            break


def find_goal():
    kvasir = [0.909, 0.871, 0.907, 0.926, 0.948, 0.897, 0.913, 0.914, 0.926, 0.913, 0.919, 0.959, 0.902, 0.910, 0.9,
              0.782, 0.921, 0.927, 0.898, 0.950, 0.918, 0.918, 0.939, 0.924, 0.912, 0.917, 0.917, 0.907, 0.904, 0.912,
              0.898, 0.913, 0.813, 0.821, 0.818]
    kvasir.sort()
    print(kvasir)
    print('Kvasir best mdice {}'.format(max(kvasir)))

    clinicdb = [0.89, 0.766, 0.942, 0.943, 0.945, 0.923, 0.919, 0.93, 0.934, 0.935, 0.921, 0.928, 0.91, 0.579, 0.926,
                0.936, 0.946, 0.947, 0.936, 0.918, 0.947, 0.932, 0.932, 0.916, 0.937, 0.921, 0.916, 0.926, 0.899, 0.796,
                0.794, 0.823]
    clinicdb.sort()
    print(clinicdb)
    print('CVC-ClinicDB best mdice {}'.format(max(clinicdb)))

    colondb = [0.744, 0.834, 0.825, 0.837, 0.785, 0.774, 0.797, 0.788, 0.788, 0.867, 0.894, 0.83, 0.73, 0.468, 0.814,
               0.824, 0.935, 0.772, 0.819, 0.811, 0.731, 0.777, 0.808, 0.755, 0.753, 0.709, 0.483, 0.512]
    colondb.sort()
    print(colondb)
    print('CVC-ColonDB best mdice {}'.format(max(colondb)))

    t = [0.932, 0, 0, 0.905, 0.924, 0, 0.909, 0.894, 0.909, 0.893, 0.9, 0.924, 0.924, 0.726, 0.726, 0.905, 0.905, 0.903,
         0.903, 0.906, 0.906, 0.887, 0.892, 0.9, 0.869, 0.888, 0.871, 0.866, 0.831, 0.707, 0.71]
    t.sort()
    print(t)
    print('CVC-300 best mdice {}'.format(max(t)))

    etis = [0.743, 0.823, 0.801, 0.777, 0.749, 0.759, 0.726, 0.78, 0.78, 0.903, 0.797, 0.551, 0.788, 0.823, 0.935,
            0.747, 0.842, 0.801, 0.677, 0.762, 0.787, 0.719, 0.75, 0.766, 0.628, 0.401, 0.398]
    etis.sort()
    print(etis)
    print('ETIS best mdice {}'.format(max(etis)))

    bkai = [0.66, 0.902]
    print('BKAI best mdice {}'.format(max(bkai)))

    # BKAI: 0.902
    # CVC-300: 0.932
    # CVC-ClinicDB: 0.947
    # CVC-ColonDB: 0.935
    # ETIS: 0.935
    # Kvasir: 0.959


def retest_failed_cases():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=int, default=352, help='testing image size')
    parser.add_argument('--save_path', type=str, default='./snapshot/ChocolateNet.pth')
    args = parser.parse_args()

    model = ChocolateNet1()
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    model.cuda()

    # an experiment for discovery ba_output, sa_output, fa_output
    failed_cases = {'BKAI-IGH-NEOPOLYP': [14, 819, 853, 945, 951, 568, 492, 342], 'CVC-ClinicDB': [46],
                    'CVC-ColonDB': [111, 198, 220, 233, 239, 282, 290],
                    'ETIS-LaribPolypDB': [149, 126, 164, 166, 168, 104]}
    for testset_name in ['BKAI-IGH-NEOPOLYP', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        data_path = './dataset/testset/{}'.format(testset_name)
        image_path = '{}/images/'.format(data_path)
        mask_path = '{}/masks/'.format(data_path)
        test_set = TestSet(image_path, mask_path, args.test_size)

        for i in failed_cases[testset_name]:
            image, mask, name = test_set.get_data(i)
            image = image.cuda()
            model(image)
    # an experiment for discovery ba_output, sa_output, fa_output


def detect_all_black():
    failed_cases = {}
    dir = 'D:\\Study\\PostgraduateStudy\\2024autumn\\chocolatenet_preds\\20240907\\PCS\\'

    for dataset in ['BKAI-IGH-NEOPOLYP', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        failed_cases[dataset] = []
        preds = [dir + dataset + '\\' + pred for pred in os.listdir(dir + dataset) if pred.endswith('.png')]

        for pred in preds:
            p = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
            if (p == 0).all():
                failed_cases[dataset].append(pred.split('\\')[-1])

    print(failed_cases)


def normalize_positive_pixels(tensor):
    pos_mask = tensor > 0
    pos_values = tensor[pos_mask]
    a = pos_values.float().min()
    b = pos_values.float().max()
    normalized_pos_values = (pos_values - a) / (b - a)
    tensor[pos_mask] = normalized_pos_values
    return tensor


def show_boundary(pred, save_path, name):
    pos_min = pred[pred > 0].float().min()
    pos_max = pred[pred > 0].float().max()
    pred[pred > 0] = (pred[pred > 0] - pos_min) / (pos_max - pos_min + 1e-8)
    neg_min = pred[pred < 0].float().min()
    neg_max = pred[pred < 0].float().max()
    pred[pred < 0] = (pred[pred < 0] - neg_min) / (neg_max - neg_min + 1e-8)
    pred = np.round(pred.data.cpu().numpy() * 255)
    cv2.imwrite(save_path + name, pred)


def test_new_transforms():
    transform = transforms.Compose([
        transforms.RandomRotation(90, expand=False, center=None, fill=0),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Resize((352, 352)),
        # TODO: 补充其他可用的增强方法，比如亮度，对比度，染色
        # transforms.ColorJitter(brightness=(1, 1.5), contrast=0, saturation=0, hue=(-0.1, 0.1)),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)),
        # reference from ColonFormer
        # # 高斯模糊，标准差0.001-2.0
        # transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
        # # 图像的亮度调整为0.4，对比度调整为0.5，饱和度调整为0.25，色度调整为0.01
        # transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = SegmentationDataset(transform=transform)
    for i in range(10):
        image = dataset.get_image(i)
        # img0 = np.array(image)
        # cv2.imshow('before aug', img0)
        # cv2.waitKey(0)
        image = transform(image)
        img = image.permute(1, 2, 0).data.cpu().numpy()
        cv2.imshow('after aug', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # progress_bar()

    experiment_of_dye()

    # process_bkai_dataset()

    # calc_mean_std()

    # sum_dict = count_polyp()

    # sum_dict = {'BKAI-IGH-NEOPOLYP': [1000, 744, 186, 63, 7], 'CVC-300': [60, 51, 7, 2, 0],
    #             'CVC-ClinicDB': [612, 106, 219, 194, 93], 'CVC-ColonDB': [380, 215, 93, 42, 30],
    #             'ETIS-LaribPolypDB': [196, 151, 27, 18, 0], 'Kvasir': [1000, 167, 317, 339, 177],
    #             'all': [3248, 1434, 849, 658, 307]}
    # draw_graph(sum_dict)

    # calc_model_complexity()  # ChocolateNet's flops = 10.48 GMac and its params = 25.12 M(2024/04/15 20:41)

    # calc_fps(10)

    # test_boundary_attention()

    # test_structure_attention()

    # test_feature_aggregation()

    # find_goal()

    # retest_failed_cases()

    # detect_all_black()

    # print(normalize_positive_pixels(torch.tensor([[-1., 2., -3., 4.], [5., -6., 7., -8.]])))

    # show_boundary()

    # test_new_transforms()
