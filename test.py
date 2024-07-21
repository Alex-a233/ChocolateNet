import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm

from model import ChocolateNet
from utils.dataloader import TestSet
from utils.useful_func import empty_create, calculate_time_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=int, default=352, help='testing image size')
    parser.add_argument('--save_path', type=str, default='./snapshot/ChocolateNet.pth')
    args = parser.parse_args()

    model = ChocolateNet()
    model.load_state_dict(torch.load(args.save_path))
    model.cuda()
    model.eval()

    start_time = datetime.now()
    print('$' * 20, 'Testing start and it is time about {}'.format(start_time), '$' * 20)

    for testset_name in ['BKAI-IGH-NEOPOLYP', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
        print('$' * 15, 'Processing {} start'.format(testset_name), '$' * 15)
        data_path = './dataset/testset/{}'.format(testset_name)
        save_path = './predicts/ChocolateNet/{}/'.format(testset_name)

        empty_create(save_path)
        image_path = '{}/images/'.format(data_path)
        mask_path = '{}/masks/'.format(data_path)
        num_mask = len(os.listdir(mask_path))
        test_set = TestSet(image_path, mask_path, args.test_size)

        for i in tqdm(range(num_mask), colour='#e946ef'):
            image, mask, name = test_set.load_data()
            mask = np.asarray(mask, np.float32)
            mask /= (mask.max(initial=.0) + 1e-8)  # TODO: beware here, max(initial = 0) maybe wrong
            image = image.cuda()

            pred = model(image)
            pred = F.interpolate(pred, size=mask.shape, mode='bilinear', align_corners=False)[0, 0]
            # refers from PCS in SANet，enforce the contrast between pos & neg samples
            pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
            pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
            pred = pred.sigmoid().data.cpu().numpy()

            empty_create(save_path + name)
            cv2.imwrite(save_path + name, pred * 255)

        print('$' * 15, 'Processing {} end'.format(testset_name), '$' * 15)

    end_time = datetime.now()
    print('$' * 20, 'Testing end and it is time about {}'.format(end_time), '$' * 20 + '\n')

    calculate_time_loss(start_time, end_time, 'Testing')
