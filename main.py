# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, transforms
from tqdm import tqdm


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name} ^_^y')  # Press Ctrl+F8 to toggle the breakpoint.


def progress_bar():
    num_mask = 100
    for i in tqdm(range(num_mask), colour='#e946ef'):
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


def cpr_colormap(img_path, dict):
    h1 = []
    h2 = []
    origin_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h1.append(origin_img)
    h2.append(origin_img)
    # cv2.imshow('origin_image', origin_img)
    img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    i = 0
    for k, v in dict.items():
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
    bkai_path = './dataset/testset/BKAI-IGH-NEOPOLYP/masks/'
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
    # train_imgs = [os.path.join('./dataset/trainset/images', img) for img in os.listdir('./dataset/trainset/images') if img.endswith('.png')]
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('ChocolateNet')

    # progress_bar()

    # experiment_of_dye()

    # process_bkai_dataset()

    calc_mean_std()  # TODO: wait for author reply, if i guess right, change norm
