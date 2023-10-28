# -*- encoding: utf-8 -*-
#Time        :2021/03/02 19:42:54
#Author      :Chen
#FileName    :train.py
#Version     :1.0

import torch
import _init_paths
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.metric import Metrics, setup_seed, test
from utils.metric import evaluate
from utils.loss import BceDiceLoss, BCELoss, DiceLoss
from datasets.polyp_dataset import get_data_aug
from nets.pranet import CRANet
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 8
NUM_WORKERS = 4
POWER = 0.9
INPUT_SIZE = (256, 256)
DATA_ROOT = '/home/cyang/EndoCV/Data'

LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 150
VALID_STEPS = 100
GPU = '0'
RESTORE_FROM = '/home/cyang/SFDA/checkpoint/EndoScene.pth'
SNAPSHOT_DIR = '/home/cyang/EndoCV/checkpoint'
SAVE_RESULT = False
RANDOM_MIRROR = True

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, length, power=0.9):
    lr = lr_poly(args.learning_rate, i_iter, NUM_STEPS * length, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_ROOT,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--valid-steps", type=int, default=VALID_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", type=bool, default=RANDOM_MIRROR,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=SAVE_RESULT,
                        help="Whether to save the predictions.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    return parser.parse_args()


args = get_arguments()


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_loader, test_loader = get_data_aug(
        data_root = args.data_dir,
        batch_size = args.batch_size
    )

    model = CRANet().cuda()
    #model.load_state_dict(torch.load(args.restore_from))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    source_criterion = BCELoss()
    #target_criterion = criterion.BCELoss()
    Best_dice = 0
    for epoch in range(args.num_steps):
        seg_loss = 0
        tic = time.time()
        model.train()
        for i_iter, batch in enumerate(train_loader):
            data, name = batch
            image = data['image']
            label = data['label']

            image = Variable(image).cuda()

            label = Variable(label).cuda()
            gts = label.unsqueeze(1)

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(image)

            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            seg_loss += loss.item()
            lr = adjust_learning_rate(optimizer=optimizer, i_iter=i_iter + len(train_loader), length=len(train_loader))
            #lr = args.learning_rate
        batch_time = time.time() - tic
        print('Epoch: [{}/{}], Time: {:.2f}, '
              'lr: {:.6f}, Loss: {:.6f}' .format(
                  epoch, args.num_steps, batch_time, lr, seg_loss))
        # begin test on target domain
        dice = test(model, test_loader, args)
        if Best_dice <= dice:
            Best_dice = dice
            torch.save(model.state_dict(), '/home/cyang/EndoCV/checkpoint/pranet_best.pth')
    torch.save(model.state_dict(), '/home/cyang/EndoCV/checkpoint/pranet_last.pth')

if __name__ == '__main__':
    main()
