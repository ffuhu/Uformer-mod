# test_in_any_resolution_stack.py
# --input_dir
# /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good
# --result_dir
# /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/results_Uformerdelur_SGgenMVDS_FT_MVDSmanually_selected_5_2/
# --weights
# /home/felix/Scratch/nus/20_deblurring_uformer/log/Uformerdelur_SGgenMVDS_FT_MVDSmanually_selected_5_2/models/model_best.pth
# --embed_dim
# 16
# --eval_workers
# 0
# --train_ps
# 64
# --save_images



import numpy as np
import os, sys, math
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


parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', required=True, type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', required=True, type=str, help='Directory for results')
parser.add_argument('--weights', required=True, type=str, help='Path to weights')
parser.add_argument('--ext', default='.tif', type=str, help='file extension to look for')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

# args for vit
parser.add_argument('--in_chans', type=int, default=1, help='vit input channels (1 for grayscale, 3 for RGB)')
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

# utils.mkdir(args.result_dir)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

with torch.no_grad():

    stack_list = glob.glob(args.input_dir + os.sep + '*' + args.ext)
    for stack_path in stack_list:

        img_stack_blurry = tifffile.imread(stack_path)
        img_stack_restored = np.zeros_like(img_stack_blurry)

        psnr_val_rgb = []
        ssim_val_rgb = []
        for stack_idx in tqdm(range(img_stack_blurry.shape[0])):
            stack_path_dir = os.path.dirname(stack_path)
            stack_path_name = os.path.basename(stack_path)

            img_blurry = img_stack_blurry[stack_idx].astype(np.float32)
            img_blurry_shape = img_blurry.shape
            img_blurry_min = img_blurry.min()
            img_blurry_max = img_blurry.max()

            img_blurry = (img_blurry - img_blurry_min) / (img_blurry_max - img_blurry_min)
            img_blurry = torch.from_numpy(img_blurry[np.newaxis, np.newaxis])

            # rgb_gt = data_test[0].numpy()[0].transpose((1, 2, 0))
            # The factor is calculated (window_size(8) * down_scale(2^4) in this case)
            img_blurry_sq, mask = expand2square(img_blurry.cuda(), factor=128, n_channels=args.in_chans)
            img_restored = model_restoration(img_blurry_sq, 1 - mask)

            img_restored = torch.masked_select(img_restored, mask.bool()).reshape(1, args.in_chans, img_blurry_shape[0], img_blurry_shape[1])
            img_restored = torch.clamp(img_restored, 0, 1).cpu().numpy()[0].transpose((1, 2, 0)).squeeze()
            img_blurry = img_blurry.cpu().numpy()[0].transpose((1, 2, 0)).squeeze()

            img_stack_restored[stack_idx] = img_restored * (img_blurry_max - img_blurry_min) + img_blurry_min

            # psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
            # ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))
            psnr_val_rgb.append(psnr_loss(img_blurry, img_restored))
            ssim_val_rgb.append(ssim_loss(img_blurry, img_restored, multichannel=True))

            # if args.save_images:
            #     utils.save_img(os.path.join(args.result_dir, filenames[0]), img_as_ubyte(img_restored))
            #     fn_ext = os.path.splitext(filenames[0])[-1]
            #     filename_noisy = filenames[0].replace(fn_ext, '_blurry' + fn_ext)
            #     utils.save_img(os.path.join(args.result_dir, filename_noisy), img_as_ubyte(img_blurry))

        if args.save_images:
            exp_name = args.weights.split(os.sep)[-3]
            stack_path_dir_to_save = os.path.join(stack_path_dir, 'deblurred_' + exp_name)
            os.makedirs(stack_path_dir_to_save, exist_ok=True)
            stack_restored_path = stack_path_name.replace(args.ext, f'_restored_UFormer{args.ext}')
            tifffile.imwrite(os.path.join(stack_path_dir_to_save, stack_restored_path), img_stack_restored)

    psnr_val_rgb = sum(psnr_val_rgb) / img_stack_blurry.shape[0]
    ssim_val_rgb = sum(ssim_val_rgb) / img_stack_blurry.shape[0]
    print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
