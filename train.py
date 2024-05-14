import os
from config import Config

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from abla_exp.CSBNet import CSBNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from tensorboardX import SummaryWriter

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
# model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = CSBNet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_L1 = nn.L1Loss()
criterion_ssim = losses.SSIM_loss()
# criterion = losses.CharbonnierLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')
utils.print_network(model_restoration)

best_psnr = 0
best_ssim = 0
best_psnr_epoch = 0
best_ssim_epoch = 0
writer = SummaryWriter('logs/only_ED')
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored = model_restoration(input_)

        # Compute loss at each stage
        # res = torch.clamp(restored, 0, 1)
        res_1 = torch.clamp(restored[0], 0, 1)
        res_2 = torch.clamp(restored[1], 0, 1)
        res_3 = torch.clamp(restored[2], 0, 1)

        target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
        target_4 = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)

        # loss = criterion_L1(res_1, target) + criterion_L1(res_2, target_2) + criterion_L1(res_3, target_4) + \
        #        (1 - criterion_ssim(res_1, target)) + (1 - criterion_ssim(res_2, target_2)) + (1 - criterion_ssim(res_3, target_4))
        loss = (1 - criterion_ssim(res_1, target)) + (1 - criterion_ssim(res_2, target_2)) + (1 - criterion_ssim(res_3, target_4))

        # loss = criterion(res_1, target) + criterion(res_2, target_2) + criterion(res_3, target_4) + \
        #        (1 - criterion_ssim(res_1, target)) + (1 - criterion_ssim(res_2, target_2)) + (1 - criterion_ssim(res_3, target_4))
        # loss = criterion(res_1, target) + criterion(res_2, target) + criterion(res_3, target)
        # loss = criterion_L1(res, target) + (1 - criterion_ssim(res, target))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]

            for res, tar in zip(restored, target):
                res = torch.clamp(res, 0, 1)
                res = res.cpu().numpy().transpose(1, 2, 0)
                tar = tar.cpu().numpy().transpose(1, 2, 0)
                psnr_val_rgb.append(compare_psnr(tar, res, data_range=1))
                ssim_val_rgb.append(compare_ssim(tar, res, gaussian_weights=True, multichannel=True, data_range=1))

        psnr_val_rgb = np.stack(psnr_val_rgb).mean()
        ssim_val_rgb = np.stack(ssim_val_rgb).mean()
        writer.add_scalar('psnr', psnr_val_rgb, epoch)
        writer.add_scalar('ssim', ssim_val_rgb, epoch)

        writer.add_scalar('loss', epoch_loss / len(train_loader), epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_psnr_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_psnr_best.pth"))

        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_ssim_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_ssim_best.pth"))

        print(
            "[epoch %d PSNR: %.4f SSIM: %.4f --- best_psnr_epoch %d Best_PSNR %.4f --- best_ssim_epoch %d Best_SSIM %.4f]"
            % (epoch, psnr_val_rgb, ssim_val_rgb, best_psnr_epoch, best_psnr, best_ssim_epoch, best_ssim))

        # torch.save({'epoch': epoch,
        #             'state_dict': model_restoration.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss / len(train_loader),
                                                                              scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

