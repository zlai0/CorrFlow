import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np

import functional.feeder.dataset.Davis2017 as D
import functional.feeder.dataset.DavisLoader as DL
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from models.corrflow import CorrFlow
from functional.utils.io import imwrite_indexed

import logger


def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData = D.dataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.myImageFloder(TrainData[0], TrainData[1], False),
        batch_size=1, shuffle=False,num_workers=0,drop_last=False
    )

    model = CorrFlow(args)
    model = nn.DataParallel(model).cuda()

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    test(TrainImgLoader, model, log)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def test(dataloader, model, log):
    model.eval()

    Fs = AverageMeter()
    Js = AverageMeter()

    n_b = len(dataloader)

    log.info("Start testing.")
    for b_i, (images_rgb, annotations) in enumerate(dataloader):
        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]

        N = len(images_rgb)

        for i in range(N-1):
            rgb_0 = images_rgb[i]
            rgb_1 = images_rgb[i+1]
            if i == 0:
                anno_0 = annotations[i]
            else:
                anno_0 = output
            anno_1 = annotations[i+1]

            _, _, h, w = anno_0.size()

            with torch.no_grad():
                output = model(rgb_0, anno_0, rgb_1)
                output = F.interpolate(output, (h,w), mode='bilinear')
                output = torch.argmax(output, 1, keepdim=True).float()

            max_class = anno_1.max()
            js, fs = [], []

            for classid in range(1, max_class + 1):
                obj_true = (anno_1 == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fs.append(f); js.append(j)

            ###
            folder = os.path.join(args.savepath,'benchmark')
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, D.catnames[b_i].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad = ((0,0),(3,3)) if anno_0.size(3) < 1152 else ((0,0), (0,0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
                out_img = anno_0[0, 0].cpu().numpy().astype(np.uint8)

                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img )

            output_file = os.path.join(output_folder, '%s.png' % str(i + 1).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

            f = np.mean(fs); j = np.mean(js)
            Fs.update(f); Js.update(j)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).'
                          .format(Js.avg, Fs.avg)])

        log.info('[{}/{}] {}'.format( b_i, n_b, info ))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CorrFlow')

    # Data options
    parser.add_argument('--datapath', help='Data path for Davis')
    parser.add_argument('--savepath', type=str, default='results/test',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')

    args = parser.parse_args()

    main()