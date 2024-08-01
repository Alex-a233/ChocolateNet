import argparse
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ChocolateNet
from utils.dataloader import TrainSet
from utils.loss_func import wbce_wdice
from utils.useful_func import empty_create, choose_best, print_save, calculate_time_loss


def train(model, trainset_loader, args):  # sup,
    total_step = len(trainset_loader)
    scaler = amp.GradScaler(enabled=True)
    logger = SummaryWriter(args.log_path)
    best_mdice = 0.
    early_stopping_cnt = 0
    eval_epoch = args.epoch // 3
    global_step = 1
    size_rates = [0.5, 0.75, 1, 1.25, 1.5]  # TODO: this maybe affects training speed

    for epoch in range(1, args.epoch + 1):
        params = model.parameters()
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=args.weight_decay)

        best_before = best_mdice
        for step, (images, masks) in enumerate(trainset_loader, start=1):
            images = images.cuda().float()
            masks = masks.cuda().float()

            # for size_rate in size_rates: # TODO: comment this temporarily
            size_rate = np.random.choice(size_rates, p=[.1, .2, .4, .2, .1])

            if size_rate != 1.0:
                adjusted_size = int(round(args.train_size * size_rate / 32) * 32)
                images = F.interpolate(images, size=(adjusted_size, adjusted_size), mode='bilinear',
                                       align_corners=True)
                masks = F.interpolate(masks, size=(adjusted_size, adjusted_size), mode='bilinear',
                                      align_corners=True)

            optimizer.zero_grad()
            # use amp
            with amp.autocast():
                pred = model(images)
                # calculate total loss TODO: calculate 3 loss, means cost ...
                bce_loss_2, dice_loss_2 = wbce_wdice(pred, masks)
                # sup_loss = sup(preds[2], masks)
                loss = (bce_loss_2 + dice_loss_2).mean()
                # + 0.1 * sup_loss

            try:
                # backward propagation
                scaler.scale(loss).backward()
            except Exception:
                ex = traceback.format_exc()
                print_save(f'it is time about {datetime.now()} exception occur =>\n{ex}', args.log_path,
                           args.log_name)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            # record the changes of learning rate & total loss
            logger.add_scalars('lr', {'lr': optimizer.param_groups[0]['lr']}, global_step=global_step)
            # 'sup_loss': sup_loss.mean().item(),
            logger.add_scalars('loss', {'bce': bce_loss_2.mean().item(), 'dice': dice_loss_2.mean().item(),
                                        'total_loss': loss.item()}, global_step=global_step)

            if step % 10 == 0 or step == total_step:
                print_save('current time {}, epoch [{:03d}/{:03d}], step [{:04d}/{:04d}], loss:{:04f}]'.format(
                    datetime.now(), epoch, args.epoch, step, total_step, loss), args.log_path, args.log_name)

        # eval current model begin from 3rd of total_epochs
        if epoch >= eval_epoch:
            mean_dice, record = choose_best(model, args)
            # save the best model weights args if cur mdice better than ever
            if mean_dice > best_mdice:
                best_mdice = mean_dice
                # create save path
                empty_create(args.save_path)
                torch.save(model.state_dict(), args.save_path + 'ChocolateNet' + str(epoch) + '.pth')
                print_save('current time %s, epoch %s & best model\'s mdice %s' % (datetime.now(), epoch, best_mdice),
                           args.log_path, args.log_name)

                for k, v in record.items():
                    print_save('dataset: {} & mdice: {:04f}'.format(k, v), args.log_path, args.log_name)

                # if there generates a better best mdice, set early_stopping_cnt to 0
                early_stopping_cnt = 0

            if best_mdice < best_before:
                early_stopping_cnt += 1

            if early_stopping_cnt == args.early_stopping_patience:
                stop_training_prompt = 'current time {}, epoch [{:03d}/{:03d}] & best mdice {:04f}. ' \
                                       'model can not perform better, stop training~'
                print_save(stop_training_prompt.format(datetime.now(), epoch, args.epoch, best_mdice), args.log_path,
                           args.log_name)
                break

    logger.close()


if __name__ == '__main__':
    cur_date = datetime.now().date().isoformat()

    parser = argparse.ArgumentParser(description='here is training arguments')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='patience for epoch number')
    parser.add_argument('--use_aug', type=bool, default=True, help='use data augmentation or not')
    parser.add_argument('--train_size', type=int, default=352, help='training image size')
    parser.add_argument('--eval_size', type=int, default=352, help='evaluating image size')
    parser.add_argument('--train_path', type=str, default='./dataset/trainset/', help='training set path')
    parser.add_argument('--eval_path', type=str, default='./dataset/testset/', help='test path, eval the best weights')
    parser.add_argument('--save_path', type=str, default='./snapshot/', help='training weights save path')
    parser.add_argument('--log_path', type=str, default='./log/{}/'.format(cur_date), help='training log save path')
    parser.add_argument('--log_name', type=str, default='training.log', help='training log\'s name')
    args = parser.parse_args()

    # this code will affect performance of the model, activating it when debug mode only ...
    # torch.autograd.set_detect_anomaly(True)

    model = ChocolateNet().cuda()
    # sup = Supervisor().cuda()

    model.train()
    # sup.eval()

    image_path = '%simages' % args.train_path
    mask_path = '%smasks' % args.train_path

    train_set = TrainSet(args)
    trainset_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)

    start_time = datetime.now()
    print_save('$' * 20 + 'Training start and it is time about {}'.format(start_time) + '$' * 20, args.log_path,
               args.log_name)

    # train(model, sup, trainset_loader, args)
    train(model, trainset_loader, args)

    end_time = datetime.now()
    print_save('$' * 20 + 'Training end and it is time about {}'.format(end_time) + '$' * 20, args.log_path,
               args.log_name)

    calculate_time_loss(start_time, end_time, 'Training')
