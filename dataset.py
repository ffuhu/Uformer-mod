import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, is_tiff_file, load_img, load_img_tiff, load_img_png, Augment_RGB_torch
import torch.nn.functional as F
import random



def reorder_filenames(filenames_to_sort, reference, token='_down'):
    filenames_sorted = []
    for fn_ref in reference:
        pattern_to_find = fn_ref.split(token)[-1]
        fn_to_sort = list(filter(lambda fn: pattern_to_find in fn, filenames_to_sort))
        assert len(fn_to_sort) == 1, 'Sorting function FAILED! Look into your train/test folder and dataset.py ' \
                                     f'(using token: {token})'
        filenames_sorted.append(fn_to_sort[0])
    return filenames_sorted


augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def _sample_random_patch(image_clean, image_blurry, patch_shape):
    # crop a stack_patch with shape self._patch_shape
    c, w, h = image_clean.shape
    cp, wp, hp = patch_shape
    ci = np.random.randint(0, c - cp)
    xi = np.random.randint(0, w - wp)
    yi = np.random.randint(0, h - hp)
    patch_clean = image_clean[ci:ci + cp, xi:xi + hp, yi:yi + wp]
    patch_blurry = image_blurry[ci:ci + cp, xi:xi + hp, yi:yi + wp]
    return patch_clean, patch_blurry

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        self.root_folder = rgb_dir
        self.img_options = img_options

        gt_dir = rgb_dir + 'clean/train/'
        input_dir = rgb_dir + 'blurry/train/'
        print('train dataloader')
        print('using gt_dir: ', gt_dir)
        print('using input_dir: ', input_dir)

        clean_files = sorted(os.listdir(gt_dir))
        noisy_files = sorted(os.listdir(input_dir))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_tiff_file(x) or is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_tiff_file(x) or is_png_file(x)]

        self.noisy_filenames = reorder_filenames(self.noisy_filenames, reference=self.clean_filenames,
                                                 token=self.img_options['token'])

        multiplier = self.img_options['data_multiplier']
        print(f'Careful! Multiplying train dataset by x{multiplier}!')
        self.clean_filenames = self.clean_filenames * multiplier
        self.noisy_filenames = self.noisy_filenames * multiplier

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        if np.random.uniform(0, 1) < self.img_options['sharpsharp']:
            # load a clean image as noisy to train clean->clean
            noisy = clean.copy()
        else:
            noisy = np.float32(load_img(self.noisy_filenames[tar_index]))

        n_channels = self.img_options['in_channel']

        suffix = ''
        if clean.ndim < 3:
            clean = np.repeat(clean[..., np.newaxis], axis=2, repeats=n_channels)
            noisy = np.repeat(noisy[..., np.newaxis], axis=2, repeats=n_channels)
        else:
            if clean.shape[-1] > n_channels:
                ini_channel = np.random.randint(0, clean.shape[-1] - n_channels)
                end_channel = ini_channel + n_channels
                clean = clean[..., ini_channel:end_channel]
                noisy = noisy[..., ini_channel:end_channel]
                suffix = f'_ic_{ini_channel}_ec{end_channel}'
            elif clean.shape[-1] == n_channels:
                suffix = f'_ic_0_ec{clean.shape[-1]}'
            else:
                raise NotImplementedError

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        fn_ext = os.path.splitext(self.clean_filenames[tar_index])[-1]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1].replace(fn_ext, suffix + fn_ext)
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1].replace(fn_ext, suffix + fn_ext)

        # Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]

        r = 0 if H - ps == 0 else np.random.randint(0, H - ps)
        c = 0 if H - ps == 0 else np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################

class DataLoaderTrain_Gaussian(Dataset):
    def __init__(self, rgb_dir, noiselevel=5, img_options=None, target_transform=None):
        super(DataLoaderTrain_Gaussian, self).__init__()

        self.target_transform = target_transform
        # pdb.set_trace()
        clean_files = sorted(os.listdir(rgb_dir))
        # noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        # clean_files = clean_files[0:83000]
        # noisy_files = noisy_files[0:83000]
        self.clean_filenames = [os.path.join(rgb_dir, x) for x in clean_files if is_png_file(x)]
        # self.noisy_filenames = [os.path.join(rgb_dir, 'input', x)       for x in noisy_files if is_png_file(x)]
        self.noiselevel = noiselevel
        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        print(self.tar_size)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        # print(self.clean_filenames[tar_index])
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # noiselevel = random.randint(5,20)
        noisy = clean + np.float32(np.random.normal(0, self.noiselevel, np.array(clean).shape) / 255.)
        noisy = np.clip(noisy, 0., 1.)

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        self.root_folder = rgb_dir
        self.img_options = img_options

        gt_dir = rgb_dir + 'clean/test/'
        input_dir = rgb_dir + 'blurry/test/'
        print('val dataloader')
        print('using gt_dir: ', gt_dir)
        print('using input_dir: ', input_dir)

        clean_files = sorted(os.listdir(gt_dir))
        noisy_files = sorted(os.listdir(input_dir))

        self.clean_filenames = [os.path.join(gt_dir, x) for x in clean_files if is_tiff_file(x) or is_png_file(x)]
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_tiff_file(x) or is_png_file(x)]

        self.noisy_filenames = reorder_filenames(self.noisy_filenames, reference=self.clean_filenames,
                                                 token=self.img_options['token'])

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))

        n_channels = self.img_options['in_channel']

        suffix = ''
        if clean.ndim == 2:
            clean = np.repeat(clean[..., np.newaxis], axis=2, repeats=n_channels)
            noisy = np.repeat(noisy[..., np.newaxis], axis=2, repeats=n_channels)
        else:
            if clean.shape[-1] > n_channels:
                ini_channel = np.random.randint(0, clean.shape[-1] - n_channels)
                end_channel = ini_channel + n_channels
                clean = clean[..., ini_channel:end_channel]
                noisy = noisy[..., ini_channel:end_channel]
                suffix = f'_ic_{ini_channel}_ec{end_channel}'
            elif clean.shape[-1] == n_channels:
                suffix = f'_ic_0_ec{clean.shape[-1]}'
            else:
                raise NotImplementedError

        # h_2pow = np.log2(clean.shape[0])
        # w_2pow = np.log2(clean.shape[1])
        # if h_2pow != int(h_2pow) or w_2pow != int(w_2pow):
        #     new_h = 512  #2 **np.ceil(h_2pow).astype(np.int)
        #     new_w = 512  #2 ** np.ceil(w_2pow).astype(np.int)
        #     clean_ = clean.copy()
        #     noisy_ = noisy.copy()
        #     clean = np.zeros((new_h, new_w, n_channels), dtype=np.float32)
        #     noisy = np.zeros((new_h, new_w, n_channels), dtype=np.float32)
        #     clean[:clean_.shape[0], :clean_.shape[1]] = clean_
        #     noisy[:noisy_.shape[0], :noisy_.shape[1]] = noisy_

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        fn_ext = os.path.splitext(self.clean_filenames[tar_index])[-1]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1].replace(fn_ext, suffix + fn_ext)
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1].replace(fn_ext, suffix + fn_ext)

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2, 0, 1)

        return noisy, noisy_filename


##################################################################################################

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTestSR, self).__init__()

        self.target_transform = target_transform

        LR_files = sorted(os.listdir(os.path.join(rgb_dir)))

        self.LR_filenames = [os.path.join(rgb_dir, x) for x in LR_files if is_png_file(x)]

        self.tar_size = len(self.LR_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        LR = torch.from_numpy(np.float32(load_img(self.LR_filenames[tar_index])))

        LR_filename = os.path.split(self.LR_filenames[tar_index])[-1]

        LR = LR.permute(2, 0, 1)

        return LR, LR_filename
