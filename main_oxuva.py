import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

import functional.feeder.dataset.OxUva as O
import functional.feeder.dataset.OxUvaLoader as OL
import logger
from models.corrflow import CorrFlow
from models.submodule import one_hot
from test import test

parser = argparse.ArgumentParser(description='CorrFlow')

# Data options
parser.add_argument('--datapath', default='/scratch/local/ramdisk/zlai/oxuva/all/',
                    help='Data path for Kinetics')
parser.add_argument('--csvpath', default='datas/oxuva.csv',
                    help='Path for csv file')
parser.add_argument('--savepath', type=str, default='results/test',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default=None,
                    help='Checkpoint file to resume')

# Training options
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--worker', type=int, default=8,
                    help='number of dataloader threads')

args = parser.parse_args()

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData = O.dataloader(args.csvpath)
    TrainImgLoader = torch.utils.data.DataLoader(
        OL.myImageFloder(args.datapath, TrainData, True),
        batch_size=args.bsize, shuffle=True, num_workers=args.worker,drop_last=True
    )

    model = CorrFlow(args)
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    for epoch in range(args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainImgLoader, model, optimizer, log, epoch)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def train(dataloader, model, optimizer, log, epoch):
    _loss = AverageMeter()
    n_b = len(dataloader)

    for b_i, (images_rgb, images_quantized) in enumerate(dataloader):
        model.train()
        b_s = time.perf_counter()
        adjust_lr(optimizer, epoch, b_i, n_b)

        images_rgb = [r.cuda() for r in images_rgb]
        images_quantized = [q.cuda() for q in images_quantized]
        if not args.fullcolor:
            model.module.dropout2d(images_rgb)

        optimizer.zero_grad()


        l_sim = compute_ls(model, images_rgb, images_quantized, b_i, epoch, n_b)
        l_long = compute_ll(model, images_rgb, images_quantized)

        sum_loss = l_sim + l_long * 0.1

        sum_loss.backward()
        optimizer.step()
        _loss.update(sum_loss.item())

        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        for param_group in optimizer.param_groups:
            lr_now = param_group['lr']
        log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
            epoch, b_i, n_b, info, b_t, lr_now))

        if b_i > 0 and (b_i * args.bsize) % 10000 < args.bsize:

                log.info("Saving new checkpoint.")
                savefilename = args.savepath + '/checkpoint.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, savefilename)


def compute_ls(model, image_rgb, image_q, bi, epoch, n_b):
    eps_s, eps_e = 0.9, 0.6
    b, c, h, w = image_rgb[0].size()

    # Loss similarity
    l_sim = 0

    for i in range(2): # 3-1
        ref_g = image_rgb[i]
        tar_g = image_rgb[i + 1]
        tar_c = image_q[i + 1]
        tar_c = torch.squeeze(tar_c, 1)

        total_batch = args.epochs * n_b
        current_batch = epoch * n_b + bi
        thres = eps_s - (eps_s - eps_e) * current_batch / total_batch
        truth = np.random.random() < thres
        ref_c = image_q[i] if truth or (i == 0) else outputs

        outputs = model(ref_g, ref_c, tar_g)
        outputs = F.interpolate(outputs, (h, w), mode='bilinear')

        loss = cross_entropy(outputs, tar_c, size_average=True)
        l_sim += loss

    return l_sim

def compute_ll(model, image_rgb, image_q):
    b, c, h, w = image_rgb[0].size()
    # Loss long
    l_long = 0

    for i in range(1,3):
        for j in range(i):
            ref_g = image_rgb[j]
            if j == 0:
                ref_c = image_q[j]
            else:
                ref_c = outputs

            tar_g = image_rgb[j + 1]

            outputs = model(ref_g, ref_c, tar_g)
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')

        for j in range(i,0,-1):
            ref_g = image_rgb[j]
            ref_c = outputs
            tar_g = image_rgb[j-1]
            outputs = model(ref_g, ref_c, tar_g)
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')

        tar_c = image_q[0]
        tar_c = torch.squeeze(tar_c, 1)

        loss = cross_entropy(outputs, tar_c, size_average=True)
        l_long += loss

    return l_long


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch, batch, n_b):
    iteration = (batch + epoch * n_b) * args.bsize

    if iteration <= 400000:
        lr = args.lr
    elif iteration <= 600000:
        lr = args.lr * 0.5
    elif iteration <= 800000:
        lr = args.lr * 0.25
    elif iteration <= 1000000:
        lr = args.lr * 0.125
    else:
        lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    if size_average:
        reduction = 'mean'
    return F.nll_loss(torch.log(input + 1e-8), target, weight, None, ignore_index, None, reduction)

if __name__ == '__main__':
    main()