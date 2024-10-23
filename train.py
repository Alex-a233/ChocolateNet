import argparse
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as tvu
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ChocolateNet
from utils.dataloader import TrainSet
from utils.loss_func import wbce_wdice
from utils.useful_func import empty_create, choose_best, print_save, calculate_time_loss, clip_gradient


def train(model, trainset_loader, args):
    # size_rates = [0.5, 0.75, 1, 1.25, 1.5]
    size_rates = [0.75, 1, 1.25]
    params = model.parameters()
    optimizer = AdamW(params, args.lr, weight_decay=args.weight_decay)
    print('\\(◕ ᴗ ◕ )✿\n', optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, verbose=True, min_lr=1e-6)
    scaler = GradScaler(enabled=True)
    logger = SummaryWriter(args.log_path)
    total_step = len(trainset_loader)
    global_step = 1
    best_mdice = 0.0
    early_stopping_cnt = 0

    for epoch in range(1, args.epoch + 1):

        for step, pairs in enumerate(trainset_loader, start=1):

            for size_rate in size_rates:
                images, masks = pairs

                # if over 20 epochs the performance of model did not increase
                # if best_mdice > 0.8 and early_stopping_cnt >= 20:
                #     images = dye_aug(images)  # use dye augmentation strategy

                images = images.cuda().float()
                masks = masks.cuda().float()

                if size_rate != 1:
                    adjusted_size = int(round(args.train_size * size_rate / 32) * 32)
                    images = F.interpolate(images, size=(adjusted_size, adjusted_size), mode='bilinear',
                                           align_corners=True)
                    masks = F.interpolate(masks, size=(adjusted_size, adjusted_size), mode='bilinear',
                                          align_corners=True)

                optimizer.zero_grad()
                # use amp.autocast
                with autocast():
                    preds = model(images)
                    bce_loss, dice_loss = wbce_wdice(preds, masks)
                    # calculate total loss
                    loss = (bce_loss + dice_loss).mean()

                try:
                    scaler.scale(loss).backward()
                except Exception:
                    ex = traceback.format_exc()
                    print_save(f'it is time about {datetime.now()} exception occur =>\n{ex}', args.log_path,
                               args.log_name)

                # clip_gradient(optimizer, args.clip)  # TODO: 去掉这个会如何 ?
                scaler.step(optimizer)
                scaler.update()

                if size_rate == 1:
                    # record the changes of learning rate & total loss
                    logger.add_scalars('lr', {'lr': optimizer.param_groups[0]['lr']}, global_step=global_step)
                    logger.add_scalars('loss', {'bce': bce_loss.mean().item(), 'dice': dice_loss.mean().item(),
                                                'total_loss': loss.item()}, global_step=global_step)

            global_step += 1
            if step % 10 == 0 or step == total_step:
                print_save('current time {}, epoch [{:03d}/{:03d}], step [{:04d}/{:04d}], loss:{:04f}]'.format(
                    datetime.now(), epoch, args.epoch, step, total_step, loss), args.log_path, args.log_name)

        mean_dice, record = choose_best(model, args)

        # if mdice doesn't increase over 10 epochs, scheduler will decrease lr by multiply 5e-1
        scheduler.step(mean_dice)
        # save the best model weights args if cur mdice better than ever before
        if mean_dice > best_mdice:
            # visualized the pred
            pred_images = tvu.make_grid(preds, normalize=False, scale_each=True)
            logger.add_image('{}/preds/'.format(epoch), pred_images, epoch)
            best_mdice = mean_dice
            # create save path
            empty_create(args.save_path)
            torch.save(model.state_dict(), args.save_path + 'ChocolateNet' + str(epoch) + '.pth')
            print_save('current time %s, epoch %s & best model\'s mdice %s' % (datetime.now(), epoch, best_mdice),
                       args.log_path, args.log_name)
            # print and save performance of model
            for k, v in record.items():
                print_save('dataset: {} & mdice: {:04f}'.format(k, v), args.log_path, args.log_name)

            # if it generates a better best_mdice, reset early_stopping_cnt to 0
            early_stopping_cnt = 0
        else:
            # just_save('current time %s, epoch %s & model\'s mdice %s' % (datetime.now(), epoch, mean_dice),
            #           args.log_path, args.log_name)
            early_stopping_cnt += 1

        if early_stopping_cnt == args.early_stopping_patience:
            stop_training_prompt = 'current time {}, epoch [{:03d}/{:03d}] & best mdice {:04f}. ' \
                                   'model can not perform better, stop training~'
            print_save(stop_training_prompt.format(datetime.now(), epoch, args.epoch, best_mdice), args.log_path,
                       args.log_name)
            break

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='here is training arguments')
    parser.add_argument('--epoch', type=int, default=90, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--weight_decay', type=float, default=3e-2, help='weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='patience for epoch number')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--eval_size', type=int, default=352, help='evaluating image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    parser.add_argument('--eval_path', type=str, default='./dataset/testset/', help='test path, eval the best weights')
    parser.add_argument('--save_path', type=str, default='./snapshot/', help='training weights save path')
    cur_date = datetime.now().date()
    parser.add_argument('--log_path', type=str, default='./log/{}/'.format(cur_date), help='training log save path')
    parser.add_argument('--log_name', type=str, default='training.log', help='training log\'s name')
    args = parser.parse_args()

    # this code will affect performance of the model, activating it when debug mode only ...
    # torch.autograd.set_detect_anomaly(True)

    model = ChocolateNet().cuda()
    model.train()

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)

    # dye_aug = transforms.ColorJitter(brightness=(1, 1.5), contrast=0, saturation=0, hue=(-0.1, 0.1))

    start_time = datetime.now()
    print_save('$' * 20 + ' Training start and it is time about {} '.format(start_time) + '$' * 20, args.log_path,
               args.log_name)

    train(model, trainset_loader, args)

    end_time = datetime.now()
    print_save('$' * 20 + ' Training end and it is time about {} '.format(end_time) + '$' * 20, args.log_path,
               args.log_name)

    calculate_time_loss(start_time, end_time, 'Training', args)
