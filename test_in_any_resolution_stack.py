# test_in_any_resolution_stack.py
# --input_dir
# /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry
# --results_dir
# /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/
# --weights
# /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D/models/model_best_PSNR_SSIM.pth
# --embed_dim
# 16
# --eval_workers
# 0
# --train_ps
# 64
# --save_images
# --arch
# UNet
# --in_channel
# 1
# --out_channel
# 1


# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D
# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64

import numpy as np
import os, sys, math, re
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# sys.path.append('/home/wangzd/uformer/')

import scipy.io as sio
from utils.loader import get_validation_data
import utils
import tifffile
import glob

from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


def expand2square(timg, factor=16.0, n_channels=3):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, n_channels, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    return img, mask


def find_gt(potential_gt_path):
    blurry_name = os.path.basename(potential_gt_path)
    gt_dir = os.path.dirname(potential_gt_path)

    list_paths_clean = glob.glob(gt_dir + '/*.tif')
    list_paths_clean_wo_w = [os.path.basename(p).replace(re.search('w[0-9]{1,1}', p).group(), '') for p in list_paths_clean]
    blurry_name_wo_w = blurry_name.replace(re.search('w[0-9]{1,1}', blurry_name).group(), '')

    match = [p == blurry_name_wo_w for p in list_paths_clean_wo_w]
    clean_path = np.asarray(list_paths_clean)[match][0]
    assert os.path.isfile(clean_path), f'Could not find clean path: {clean_path}'

    return clean_path


parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', required=True, type=str, help='Directory of blurry images')
parser.add_argument('--gt_dir', default=None, type=str, help='Directory of clean images')
parser.add_argument('--results_dir', required=True, type=str, help='Directory for results')
parser.add_argument('--weights', required=True, type=str, help='Path to weights')
parser.add_argument('--ext', default='.tif', type=str, help='file extension to look for')
parser.add_argument('--pattern', default='*', type=str, help='file pattern to look for')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mixing', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

# args for vit
parser.add_argument('--arch', type=str, default='Uformer', help='architechture',
                    choices=['UNet', 'Uformer', 'MetaUformer'])
parser.add_argument('--in_channel', type=int, default=1, help='vit input channels (1 for grayscale, 3 for RGB)')
parser.add_argument('--out_channel', type=int, default=1, help='vit output channels (1 for grayscale, 3 for RGB)')
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--vit_se', action='store_true', default=False, help='SE layer')

parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
parser.add_argument('--token', type=str, default=os.sep, help='token used to check order of clean/blurry')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# utils.mkdir(args.results_dir)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []

    stack_list = glob.glob(os.path.join(args.input_dir, args.pattern + args.ext))
    for stack_path in stack_list:

        img_stack_blurry = tifffile.imread(stack_path)
        img_stack_blurry = img_stack_blurry if img_stack_blurry.ndim == 3 else img_stack_blurry[np.newaxis]
        img_stack_restored = np.zeros_like(img_stack_blurry)

        if args.gt_dir is not None:
            stack_path_gt = find_gt(potential_gt_path=stack_path.replace(args.input_dir, args.gt_dir))
            img_stack_clean = tifffile.imread(stack_path_gt)
        else:
            img_stack_clean = img_stack_blurry

        for stack_idx in tqdm(range(img_stack_blurry.shape[0]), desc=stack_path):
            stack_path_dir = os.path.dirname(stack_path)
            stack_path_name = os.path.basename(stack_path)

            if args.in_channel == 1:
                img_blurry = img_stack_blurry[stack_idx].astype(np.float32)[np.newaxis]
                img_clean = img_stack_clean[stack_idx].astype(np.float32)[np.newaxis]
            else:
                odd_n_channel = args.in_channel % 2 != 0
                offset_odd = 1 if odd_n_channel else 0
                ini_channel = np.max([stack_idx - args.in_channel // 2, 0])
                end_channel = np.min([stack_idx + args.in_channel // 2 + offset_odd, img_stack_blurry.shape[0]])
                img_blurry = img_stack_blurry[ini_channel:end_channel].astype(np.float32)
                img_clean = img_stack_clean[ini_channel:end_channel].astype(np.float32)
                channel_diff = args.in_channel - img_blurry.shape[0]

                if channel_diff > 0:
                    if ini_channel < args.in_channel:
                        fill = np.repeat(img_blurry[0][np.newaxis], channel_diff, axis=0)
                        img_blurry = np.concatenate((fill, img_blurry), axis=0)
                        img_clean = np.concatenate((fill, img_clean), axis=0)
                    else:
                        fill = np.repeat(img_blurry[-1][np.newaxis], channel_diff, axis=0)
                        img_blurry = np.concatenate((img_blurry, fill), axis=0)
                        img_clean = np.concatenate((img_clean, fill), axis=0)

            img_blurry_shape = img_blurry.shape
            img_blurry_min = img_blurry.min()
            img_blurry_max = img_blurry.max()
            img_blurry = (img_blurry - img_blurry_min) / (img_blurry_max - img_blurry_min)
            img_blurry = torch.from_numpy(img_blurry[np.newaxis])

            # rgb_gt = data_test[0].numpy()[0].transpose((1, 2, 0))
            # The factor is calculated (window_size(8) * down_scale(2^4) in this case)
            img_blurry_sq, mask = expand2square(img_blurry.cuda(), factor=128, n_channels=args.in_channel)
            img_restored = model_restoration(img_blurry_sq, 1 - mask)

            img_restored = torch.masked_select(img_restored, mask.bool()).reshape(1, args.out_channel,
                                                                                  img_blurry_shape[1],
                                                                                  img_blurry_shape[2])
            img_restored = torch.clamp(img_restored, 0, 1).cpu().numpy()[0].transpose((1, 2, 0)) #.squeeze()
            img_blurry = img_blurry.cpu().numpy()[0].transpose((1, 2, 0)) #.squeeze()

            # if args.in_channel != args.out_channel:
            #     ini_channel = img_blurry.shape[-1] // 2 - args.out_channel // 2
            #     end_channel = img_blurry.shape[-1] // 2 + args.out_channel // 2 + 1
            #     img_blurry = img_blurry[..., ini_channel:end_channel]

            # DECISION: to compute PSNR and SSIM, keep only the central slice of input and output
            if args.in_channel > 1:
                ini_channel = img_blurry.shape[-1] // 2
                end_channel = img_blurry.shape[-1] // 2 + 1
                img_blurry = img_blurry[..., ini_channel:end_channel]
                img_clean = img_clean[ini_channel:end_channel]
            if args.out_channel > 1:
                ini_channel = img_restored.shape[-1] // 2
                end_channel = img_restored.shape[-1] // 2 + 1
                img_restored = img_restored[..., ini_channel:end_channel]

            img_blurry = img_blurry.squeeze()
            img_clean = img_clean.squeeze()
            img_restored = img_restored.squeeze()

            img_stack_restored[stack_idx] = img_restored * (img_blurry_max - img_blurry_min) + img_blurry_min

            # psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
            # ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

            # psnr_val_rgb.append(psnr_loss(img_restored, img_clean))
            # ssim_val_rgb.append(ssim_loss(img_restored, img_clean, multichannel=True))

            img_restored_norm = (img_restored - img_restored.min()) / (img_restored.max() - img_restored.min())
            img_clean_norm = (img_clean - img_clean.min()) / (img_clean.max() - img_clean.min())
            psnr_val_rgb.append(psnr_loss(img_restored_norm, img_clean_norm))
            ssim_val_rgb.append(ssim_loss(img_restored_norm, img_clean_norm, multichannel=True))

            # if args.save_images:
            #     utils.save_img(os.path.join(args.results_dir, filenames[0]), img_as_ubyte(img_restored))
            #     fn_ext = os.path.splitext(filenames[0])[-1]
            #     filename_noisy = filenames[0].replace(fn_ext, '_blurry' + fn_ext)
            #     utils.save_img(os.path.join(args.results_dir, filename_noisy), img_as_ubyte(img_blurry))

        if args.save_images:
            exp_name = args.weights.split(os.sep)[-3]
            stack_path_dir_to_save = os.path.join(args.results_dir, 'deblurred_' + exp_name)
            os.makedirs(stack_path_dir_to_save, exist_ok=True)
            arch_name = 'UNet' if 'unet' in exp_name.lower() else \
                'MetaUFormer' if 'metauformer' in exp_name.lower() else \
                    'UFormer' if 'uformer' in exp_name.lower() else \
                        'UNKNOWN'
            stack_restored_path = stack_path_name.replace(args.ext, f'_restored_{arch_name}{args.ext}')
            tifffile.imwrite(os.path.join(stack_path_dir_to_save, stack_restored_path), img_stack_restored)
            stack_blurry_path_dir_to_save = os.path.join(args.results_dir, 'blurry')
            os.makedirs(stack_blurry_path_dir_to_save, exist_ok=True)
            tifffile.imwrite(os.path.join(stack_blurry_path_dir_to_save, stack_path_name), img_stack_blurry)

    psnr_val_rgb_mean = np.mean(psnr_val_rgb)
    ssim_val_rgb_mean = np.mean(ssim_val_rgb)
    psnr_val_rgb_std = np.std(psnr_val_rgb)
    ssim_val_rgb_std = np.std(ssim_val_rgb)
    print("PSNR: %f (+-%f), SSIM: %f (+-%f)" % (psnr_val_rgb_mean, psnr_val_rgb_std, ssim_val_rgb_mean, ssim_val_rgb_std))

    os.makedirs(args.results_dir, exist_ok=True)
    first_time = not os.path.exists(os.path.join(args.results_dir, 'results.txt'))
    with open(os.path.join(args.results_dir, 'results.txt'), 'a') as f:
        if first_time:
            f.write('experiment,images_deblurred,PSNR,SSIM,PSNR_std,SSIM_std\n')
        f.write(f'{exp_name},{args.input_dir},{psnr_val_rgb_mean:.6f},{ssim_val_rgb_mean:.6f},'
                f'{psnr_val_rgb_std:.6f},{ssim_val_rgb_std:.6f}\n')
