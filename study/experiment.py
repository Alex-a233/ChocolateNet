import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ptflops import get_model_complexity_info
from skimage.measure import regionprops, label, find_contours
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, transforms
from tqdm import tqdm

from backbone.pvtv2 import PvtV2B2
from model import ChocolateNet
from utils.dataloader import TrainSet
from utils.toy_block import MyConv


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
    # train_imgs = [os.path.join('./dataset/trainset/images', img) for img in os.listdir('./dataset/trainset/images')
    # if img.endswith('.png')]
    #
    # mean = [0.0, 0.0, 0.0]
    # std = [0.0, 0.0, 0.0]
    #
    # num = len(train_imgs)
    #
    # for train_img in train_imgs:
    #     img = cv2.imread(train_img, cv2.IMREAD_COLOR).astype(np.float32)
    #     img = cv2.resize(img, (352, 352))
    #     mean, std = cv2.meanStdDev(img)
    #
    #     mean[0] += mean[0]
    #     mean[1] += mean[1]
    #     mean[2] += mean[2]
    #
    #     std[0] += std[0]
    #     std[1] += std[1]
    #     std[2] += std[2]
    #
    # mean[0] /= num
    # mean[1] /= num
    # mean[2] /= num
    #
    # std[0] /= num
    # std[1] /= num
    # std[2] /= num
    #
    # print('train_set\'s \nmean = {}, \nstd = {}'.format(mean, std))

    dataset = SegmentationDataset()
    # 假设 dataset 是一个 PyTorch 的数据集
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    resize = Resize((352, 352))

    # 计算均值和方差
    mean = 0.
    std = 0.
    for img, _ in train_loader:
        img = resize(img)
        mean += img.mean([0, 2, 3])
        std += img.std([0, 2, 3])
    mean /= len(train_loader)
    std /= len(train_loader)

    print(f'Mean: {mean}')
    print(f'Std: {std}')


class SegmentationDataset(Dataset):
    def __init__(self, root_dir='./dataset/trainset/', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'masks')) if f.endswith('.png')])
        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_files[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.image_transform(mask)

        return image, mask


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


class ExpModel(nn.Module):

    def __init__(self):
        super(ExpModel, self).__init__()
        self.backbone = PvtV2B2()
        path = './pretrained_args/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # 取后三个特征图，改通道数
        self.conv2 = MyConv(128, 32, 1, is_act=False)
        self.conv3 = MyConv(320, 32, 1, is_act=False)
        self.conv4 = MyConv(512, 32, 1, is_act=False)

        self.convs3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        self.convs4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        self.convs4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        self.convs4_3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)

        self.convm3_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        self.convm4_2 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)
        self.convm4_3 = MyConv(32, 32, 3, padding=1, use_bias=True, is_act=False)

        self.conv5 = MyConv(96, 32, 3, padding=1, is_act=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
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
        x4_3_2 = self.convs4_3_2(x3_2 + x4_2 + self.up(x4_3))  # 2,3,4层异同点

        o3_2 = self.convm3_2(self.up(x3)) * x2 * x3_2
        o4_2 = self.convm4_2(self.up(self.up(x4))) * x2 * x4_2
        o4_3 = self.convm4_3(self.up(x4)) * x3 * x4_3

        res = torch.cat((self.up(o4_3), o4_2, o3_2), dim=1)
        res = self.conv5(res)
        res *= x4_3_2

        return res


def test_boundary_attention():
    model = ExpModel()
    model.cuda()
    parser = argparse.ArgumentParser(description='here is training arguments')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    args = parser.parse_args()

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    for step, (images, masks) in enumerate(trainset_loader, start=1):
        images = images.cuda().float()
        # masks = masks.cuda().float()
        res = model(images)
        print(res.shape)
        break


if __name__ == '__main__':

    # progress_bar()

    # experiment_of_dye()

    # process_bkai_dataset()

    # calc_mean_std()  # TODO: wait for author reply, if i guess right, change norm

    # sum_dict = count_polyp()

    # sum_dict = {'BKAI-IGH-NEOPOLYP': [1000, 744, 186, 63, 7], 'CVC-300': [60, 51, 7, 2, 0],
    #             'CVC-ClinicDB': [612, 106, 219, 194, 93], 'CVC-ColonDB': [380, 215, 93, 42, 30],
    #             'ETIS-LaribPolypDB': [196, 151, 27, 18, 0], 'Kvasir': [1000, 167, 317, 339, 177],
    #             'all': [3248, 1434, 849, 658, 307]}
    # draw_graph(sum_dict)

    # calc_model_complexity()  # ChocolateNet's flops = 10.48 GMac and its params = 25.12 M(2024/04/15 20:41)

    calc_fps(10)

    # test_boundary_attention()
