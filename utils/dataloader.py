import argparse
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainSet(Dataset):

    def __init__(self, args):
        self.train_size = args.train_size
        self.use_aug = args.use_aug
        image_path = os.path.join(args.train_path, 'images')
        mask_path = os.path.join(args.train_path, 'masks')
        self.images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')]
        self.masks = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.size = len(self.images)

        if self.use_aug:
            self.image_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.train_size, self.train_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet的通道级标准化参数
                # transforms.Normalize([0.497, 0.301, 0.216], [0.298, 0.208, 0.161])  # trainset 的通道级标准化系数
                # transforms.Normalize([0.496, 0.311, 0.226], [0.293, 0.210, 0.163])  # 所有 images 的通道级标准化系数(同除去BKAI的)
                transforms.Normalize([0.496, 0.336, 0.251], [0.276, 0.215, 0.165])  # ColonDB&ETIS 的通道级标准化系数
                # transforms.Normalize([0.602, 0.431, 0.372], [0.236, 0.211, 0.195])  # ETIS 的通道级标准化系数
            ])

            self.mask_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.train_size, self.train_size)),
                transforms.ToTensor()
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((self.train_size, self.train_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.497, 0.301, 0.216], [0.298, 0.208, 0.161])  # trainset 的通道级标准化系数
                # transforms.Normalize([0.496, 0.311, 0.226], [0.293, 0.210, 0.163])  # 所有 images 的通道级标准化系数
                transforms.Normalize([0.496, 0.336, 0.251], [0.276, 0.215, 0.165])  # ColonDB&ETIS 的通道级标准化系数
                # transforms.Normalize([0.602, 0.431, 0.372], [0.236, 0.211, 0.195])  # ETIS 的通道级标准化系数
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize((self.train_size, self.train_size)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')
        seed = 2024

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        image = self.image_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.size


class TestSet:

    def __init__(self, image_path, mask_path, test_size):
        self.test_size = test_size
        self.images = [image_path + f for f in os.listdir(image_path) if f.endswith('.png')]
        self.masks = [mask_path + f for f in os.listdir(mask_path) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.496, 0.311, 0.226], [0.293, 0.210, 0.163])  # 所有 images 的通道级标准化系数
            # transforms.Normalize([0.497, 0.301, 0.216], [0.298, 0.208, 0.161])  # trainset 的通道级标准化系数
            transforms.Normalize([0.496, 0.336, 0.251], [0.276, 0.215, 0.165])  # ColonDB&ETIS 的通道级标准化系数
            # transforms.Normalize([0.602, 0.431, 0.372], [0.236, 0.211, 0.195])  # ETIS 的通道级标准化系数
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        mask = Image.open(self.masks[self.index]).convert('L')
        image = self.image_transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, mask, name

    def get_data(self, i):  # for retest failed cases
        image = Image.open(self.images[i]).convert('RGB')
        mask = Image.open(self.masks[i]).convert('L')
        image = self.image_transform(image).unsqueeze(0)
        name = self.images[i].split('/')[-1]
        return image, mask, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_aug', type=bool, default=True)
    parser.add_argument('--train_size', type=int, default=352)
    parser.add_argument('--train_path', type=str, default='../dataset/trainset/')
    parser.add_argument('--eval_path', type=str, default='../dataset/testset/')
    args = parser.parse_args()
    train_set = TrainSet(args)
    print(train_set.size)
    image, mask = train_set.__getitem__(0)
    print(image.shape, mask.shape)
    test_set = TestSet(args.eval_path + 'Kvasir/images/', args.eval_path + 'Kvasir/masks/', args.train_size)
    image, mask, name = test_set.load_data()
    print(image.shape, mask.size, name)
