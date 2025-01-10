import argparse
import os

import albumentations as A
import cv2
import numpy as np
import numpy.random as random
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms as T


class TrainSet(Dataset):

    # def __init__(self, args):
    #     image_path = os.path.join(args.train_path, 'images')
    #     mask_path = os.path.join(args.train_path, 'masks')
    #     self.images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')]
    #     self.masks = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')]
    #     self.images = sorted(self.images)
    #     self.masks = sorted(self.masks)
    #     self.size = len(self.images)
    #     self.use_aug = args.use_aug
    #     self.train_size = args.train_size
    #
    #     if self.use_aug:
    #         self.image_transform = T.Compose([
    #             T.RandomRotation(90, expand=False, center=None, fill=None),
    #             T.RandomHorizontalFlip(),
    #             T.RandomVerticalFlip(),
    #             T.Resize((self.train_size, self.train_size)),
    #             T.ToTensor(),
    #             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet的通道级标准化参数
    #         ])
    #         self.mask_transform = T.Compose([
    #             T.RandomRotation(90, expand=False, center=None, fill=None),
    #             T.RandomHorizontalFlip(),
    #             T.RandomVerticalFlip(),
    #             T.Resize((self.train_size, self.train_size)),
    #             T.ToTensor()
    #         ])
    #
    #     else:
    #         self.image_transform = T.Compose([
    #             T.Resize((self.train_size, self.train_size)),
    #             T.ToTensor(),
    #             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ])
    #         self.mask_transform = T.Compose([
    #             T.Resize((self.train_size, self.train_size)),
    #             T.ToTensor()
    #         ])
    #
    # def __getitem__(self, index):
    #     image = Image.open(self.images[index]).convert('RGB')
    #     mask = Image.open(self.masks[index]).convert('L')
    #     # TODO: 染色增强策略—>息肉染色分明边界
    #
    #     seed = random.randint(2025)
    #
    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     image = self.image_transform(image)
    #
    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     mask = self.mask_transform(mask)
    #     return image, mask
    #
    # def __len__(self):
    #     return self.size

    def __init__(self, args):
        self.image_root = os.path.join(args.train_path, 'images')
        self.gt_root = os.path.join(args.train_path, 'masks')
        self.samples = [name for name in os.listdir(self.image_root) if name[0] != "."]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(args.train_size, args.train_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)

    def __getitem__(self, idx):  # TODO: try SANet's Color Exchange Strategy
        name = self.samples[idx]
        image = cv2.imread(self.image_root + '/' + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # TODO: change p = 1/2, delay first wanna test if there is no init in attn.py
        name2 = self.color1[idx % len(self.color1)] if np.random.rand() < 0.7 else self.color2[idx % len(self.color2)]
        image2 = cv2.imread(self.image_root + '/' + name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0, 1), keepdims=True), image2.std(axis=(0, 1), keepdims=True)

        image = np.uint8((image - mean) / std * std2 + mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask = cv2.imread(self.gt_root + '/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)
        pair['mask'] = torch.unsqueeze(pair['mask'], 0)
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


class TestSet:

    def __init__(self, image_path, mask_path, test_size):
        self.test_size = test_size
        self.images = [image_path + f for f in os.listdir(image_path) if f.endswith('.png')]
        self.masks = [mask_path + f for f in os.listdir(mask_path) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.image_transform = T.Compose([
            T.Resize((self.test_size, self.test_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    # print(train_set.size)
    image, mask = train_set.__getitem__(100)
    print(image.shape, mask.shape)
    test_set = TestSet(args.eval_path + 'Kvasir/images/', args.eval_path + 'Kvasir/masks/', args.train_size)
    image, mask, name = test_set.load_data()
    print(image.shape, mask.size, name)
