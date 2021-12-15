# --arch Uformer --batch_size 4 --gpu "0" --train_ps 128 --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/cropped/modified/ --env deblur_MV_of_oof_up560gaussfilt2down280 --val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/cropped/modified/ --embed_dim 16 --warmup --train_workers 8 --eval_workers 4
# --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/only_2_w4-5_21-104_0_317_150/

import os
import sys
import matplotlib.pyplot as plt


def restore_using_patches(model, img_in, patch_size=64, overlap=16):
    restored = torch.zeros_like(img_in)
    for ib in range(img_in.shape[0]):

        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0

        while xmin < img_in.shape[3]:
            ovTop = 0
            while ymin < img_in.shape[2]:
                ymin_ = min(img_in.shape[2], ymax) - patch_size
                xmin_ = min(img_in.shape[3], xmax) - patch_size
                lastPatchShiftY = ymin - ymin_
                lastPatchShiftX = xmin - xmin_
                img_in_ = img_in[ib, :, ymin_:ymax, xmin_:xmax][None, ...]
                res = model(img_in_)
                res_ = res[0, :, lastPatchShiftY:, lastPatchShiftX:][:, ovTop:, ovLeft:]
                restored[ib, :, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:] = res_.detach()

                ymin = ymin - overlap + patch_size
                ymax = ymin + patch_size
                ovTop = overlap // 2
            ymin = 0
            ymax = patch_size
            xmin = xmin - overlap + patch_size
            xmax = xmin + patch_size
            ovLeft = overlap // 2
    return restored




# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
figs_dir = os.path.join(log_dir, 'figs')
utils.mkdir(result_dir)
utils.mkdir(model_dir)
utils.mkdir(figs_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    if not opt.reset_optimizer:
        for p in optimizer.param_groups: p['lr'] = lr
        warmup = False
        new_lr = lr
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - start_epoch + 1, eta_min=1e-6)

    opt.nepoch += start_epoch

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps, 'token': opt.token, 'data_multiplier': opt.data_multiplier}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)

img_options_val = {'token': opt.token}
val_dataset = get_validation_data(opt.val_dir, img_options_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)
######### validation ###########
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        filenames = data_val[2]
        psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
    psnr_val_rgb = sum(psnr_val_rgb) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_val_rgb))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

train_losses = []
val_losses = []

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(train_loader, 0):
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        if opt.use_mixup_from_epoch != -1 and epoch >= opt.use_mixup_from_epoch:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
            loss = criterion(restored, target)
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())  # optimizer.step() happens here
        epoch_loss += loss.item()

        #### Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0:

            # save last training example processed
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(input_[0, 0].cpu()), ax[0].set_title('input')
            ax[1].imshow(target[0, 0].cpu()), ax[1].set_title('target')
            ax[2].imshow(restored[0, 0].detach().cpu()), ax[2].set_title('restored')
            plt.savefig(os.path.join(figs_dir, f'train_epoch_{epoch}_iter_{i}_img_0.png'))
            plt.close()

            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                epoch_loss_val = 0
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)

                    loss_val = criterion(restored, target)
                    epoch_loss_val += loss_val.item()

                    current_psnr = utils.batch_PSNR(restored, target, False).item()
                    psnr_val_rgb.append(current_psnr)

                    # if current_psnr > best_psnr:
                    #     fig, ax = plt.subplots(1, 3, figsize=(24, 8))
                    #     ax[0].imshow(input_[0, 0].cpu()), ax[0].set_title('input')
                    #     ax[1].imshow(target[0, 0].cpu()), ax[1].set_title('target')
                    #     ax[2].imshow(restored[0, 0].cpu()), ax[2].set_title('restored')
                    #     plt.savefig(os.path.join(figs_dir, f'epoch_{epoch}_iter_{i}_img_{ii}.png'))
                    #     # plt.imshow(input_[0, 0].cpu()), plt.savefig(os.path.join(figs_dir, f'input_epoch_{epoch}_iter_{i}.png'))
                    #     # plt.imshow(target[0, 0].cpu()), plt.savefig(os.path.join(figs_dir, f'target_epoch_{epoch}_iter_{i}.png'))
                    #     # plt.imshow(restored[0, 0].cpu()), plt.savefig(os.path.join(figs_dir, f'restored_epoch_{epoch}_iter_{i}.png'))
                    #     plt.close()

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                # plt.imshow(input_[0, 0].cpu()), plt.show()
                # plt.imshow(target[0, 0].cpu()), plt.show()
                # plt.imshow(restored[0, 0].cpu()), plt.show()

                if psnr_val_rgb > best_psnr:

                    # save images
                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        
                        with torch.cuda.amp.autocast():
                            restored = model_restoration(input_)
                            # restored_patched = restore_using_patches(model_restoration, input_, patch_size=opt.train_ps)
                        restored = torch.clamp(restored, 0, 1)
                        # restored_patched = torch.clamp(restored_patched, 0, 1)

                        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
                        # fig, ax = plt.subplots(1, 4, figsize=(32, 8))
                        ax[0].imshow(input_[0, 0].cpu()), ax[0].set_title('input')
                        ax[1].imshow(target[0, 0].cpu()), ax[1].set_title('target')
                        ax[2].imshow(restored[0, 0].cpu()), ax[2].set_title('restored')
                        # ax[3].imshow(restored_patched[0, 0].cpu()), ax[3].set_title('restored patched')
                        plt.savefig(os.path.join(figs_dir, f'epoch_{epoch}_iter_{i}_img_{ii}.png'))

                        plt.close()

                    # save model
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()

    train_losses.append(epoch_loss)
    val_losses.append(epoch_loss_val)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))

    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title('Train/val loss')
    plt.savefig(os.path.join(figs_dir, '_loss_curve.png'))
    plt.close()

    # new plot version

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epoch number')
    ax1.set_ylabel('train loss', color=color)
    ax1.plot(range(len(train_losses)), train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('val loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(val_losses)), val_losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    plt.savefig(os.path.join(figs_dir, '_loss_curve_new.png'))
    plt.close()

print("Now time is : ", datetime.datetime.now().isoformat())
