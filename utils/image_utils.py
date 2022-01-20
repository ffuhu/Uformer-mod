import os.path

import torch
import numpy as np
import pickle
import cv2
import tifffile
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_tiff_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".TIF"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


# def load_img(filepath):
#     img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img = img/255.
#     return img

def load_img(filepath):
    fn, ext = os.path.splitext(filepath)
    if ext == '.png':
        img = load_img_png(filepath)
    elif ext == '.tif':
        img = load_img_tiff(filepath)
    else:
        raise NotImplementedError
    if img.ndim > 2 and img.shape[0] < img.shape[1]:
        img = img.transpose(1, 2, 0)
    return img


def load_img_tiff(filepath):
    img = tifffile.imread(filepath)
    img = img.astype(np.float32)
    # img = img/(2**16)
    img = (img - img.min()) / (img.max() - img.min())
    # img = np.repeat(img[..., np.newaxis], axis=2, repeats=3)
    return img


def load_img_png(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    # img = img/(2**16)
    img = (img - img.min()) / (img.max() - img.min())
    # img = np.repeat(img[..., np.newaxis], axis=2, repeats=3)
    return img


def save_img(filepath, img):
    # cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(filepath, img)


def make_input_data_squared_nearest_pow2(datagen, sq_size=None):
    target = datagen[0].cuda()
    input_ = datagen[1].cuda()
    device = target.device

    h_2pow = np.log2(target.shape[-2])
    w_2pow = np.log2(target.shape[-1])
    if (h_2pow != int(h_2pow) or w_2pow != int(w_2pow)) and sq_size is not None:
        new_h = sq_size if sq_size is not None else 2 ** np.ceil(h_2pow).astype(np.int)
        new_w = sq_size if sq_size is not None else 2 ** np.ceil(w_2pow).astype(np.int)
        n_channels = target.shape[-3]
        target_ = target.clone()
        input__ = input_.clone()
        target = torch.zeros((1, n_channels, new_h, new_w), dtype=torch.float32).to(device)
        input_ = torch.zeros((1, n_channels, new_h, new_w), dtype=torch.float32).to(device)
        target[..., :target_.shape[-2], :target_.shape[-1]] = target_
        input_[..., :input__.shape[-2], :input__.shape[-1]] = input__

    return target, input_


def recover_original_dimensions(target, input_, restored, orig_shape):

    target = target[..., :orig_shape[-2], :orig_shape[-1]]
    input_ = input_[..., :orig_shape[-2], :orig_shape[-1]]
    restored = restored[..., :orig_shape[-2], :orig_shape[-1]]

    return target, input_, restored

# SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# end SSIM

def check_image_channels(img1, img2, in_channel, out_channel):
    if in_channel and out_channel and in_channel != out_channel:
        slices_x_center = in_channel // 2
        slices_x_ini = slices_x_center - out_channel // 2
        slices_x_end = slices_x_center + out_channel // 2 + 1
        if img1.shape[1] != out_channel:
            img1 = img1[:, slices_x_ini:slices_x_end]
        if img2.shape[1] != out_channel:
            img2 = img2[:, slices_x_ini:slices_x_end]
    assert img1.shape[1] == img2.shape[1], 'Images must have the same number of channels'
    return img1, img2


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def batch_PSNR(img1, img2, average=True, in_channel=None, out_channel=None):
    img1, img2 = check_image_channels(img1, img2, in_channel, out_channel)

    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR) / len(PSNR) if average else sum(PSNR)


def compute_ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def batch_SSIM(img1, img2, window_size=11, average=True, in_channel=None, out_channel=None):
    img1, img2 = check_image_channels(img1, img2, in_channel, out_channel)
    SSIM = compute_ssim(img1, img2, window_size=window_size, size_average=average)
    return sum(SSIM) / len(SSIM) if average else sum(SSIM)
