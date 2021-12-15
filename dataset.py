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


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        self.root_folder = rgb_dir
        self.img_options = img_options

        # gt_dir = 'groundtruth'
        # input_dir = 'input'
        # input_dir = rgb_dir + 'A/train/'
        # gt_dir = rgb_dir + 'B/train/'
        # input_dir = rgb_dir + 'B_blurred/train/'
        # gt_dir = rgb_dir + 'clean_medianfilter2/train/'
        # input_dir = rgb_dir + 'blur_gblur3/train/'
        # gt_dir = rgb_dir + 'of_up560gaussfilt2down280/train/'
        # # input_dir = rgb_dir + 'of_blurred_gausssigma5/train/'
        # input_dir = rgb_dir + 'oof_up560gaussfilt2down280/train/'
        gt_dir = rgb_dir + 'clean/train/'
        input_dir = rgb_dir + 'blurry/train/'
        print('train dataloader')
        print('using gt_dir: ', gt_dir)
        print('using input_dir: ', input_dir)

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        # self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        # self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_tiff_file(x) or is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_tiff_file(x) or is_png_file(x)]

        self.noisy_filenames = reorder_filenames(self.noisy_filenames, reference=self.clean_filenames,
                                                 token=self.img_options['token'])

        # # new test
        # root_folder = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/'
        # self.clean_filenames = [
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-31-18_351_434_138.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif',
        #     # root_folder + 'clean/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-28-108_14_282_138.tif',
        #     # root_folder + 'clean/210831_MultiviewTreeFrog_5_w3soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-30-362_50_137_337.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_5_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-36-93_386_336_121.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_5_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-29-123_23_336_142.tif',
        # ]
        # self.noisy_filenames = [
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-31-18_351_434_138.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif',
        #     # root_folder + 'blurry/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-28-108_14_282_138.tif',
        #     # root_folder + 'blurry/210831_MultiviewTreeFrog_5_w2soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-30-362_50_137_337.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_5_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-36-93_386_336_121.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_5_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-29-123_23_336_142.tif',
        # ]
        multiplier = self.img_options['data_multiplier']
        print(f'Careful! Multiplying train dataset by x{multiplier}!')
        self.clean_filenames = self.clean_filenames * multiplier
        self.noisy_filenames = self.noisy_filenames * multiplier

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        # clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        # clean = torch.from_numpy(np.float32(load_img_tiff(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img_tiff(self.noisy_filenames[tar_index])))

        # hack to train always with the same image

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/patches_128/5/210831_MultiviewTreeFrog_5_w2soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512_1239_uint8.png'
        # img_blurred = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/patches_128/5/210831_MultiviewTreeFrog_5_w3soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512_1239_uint8.png'
        # clean = torch.from_numpy(np.float32(load_img_png(img_clean)))
        # noisy = torch.from_numpy(np.float32(load_img_png(img_blurred)))

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif'
        # img_blurry = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif'

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif'
        # img_blurry = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif'
        # clean = np.float32(load_img_tiff(img_clean))
        # noisy = np.float32(load_img_tiff(img_blurry))

        # clean = np.float32(load_img_tiff(self.clean_filenames[tar_index]))
        # noisy = np.float32(load_img_tiff(self.noisy_filenames[tar_index]))

        clean = np.float32(load_img_png(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img_png(self.noisy_filenames[tar_index]))

        n_channels = 1
        clean = np.repeat(clean[..., np.newaxis], axis=2, repeats=n_channels)
        noisy = np.repeat(noisy[..., np.newaxis], axis=2, repeats=n_channels)

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
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

        # # gt_dir = 'groundtruth'
        # # input_dir = 'input'
        # # # input_dir = rgb_dir + 'A/test/'
        # # gt_dir = rgb_dir + 'B/test/'
        # # input_dir = rgb_dir + 'B_blurred/test/'
        # # gt_dir = rgb_dir + 'clean_medianfilter2/test/'
        # # input_dir = rgb_dir + 'blur_gblur3/test/'
        # gt_dir = rgb_dir + 'of_up560gaussfilt2down280/test/'
        # # input_dir = rgb_dir + 'of_blurred_gausssigma5/test/'
        # input_dir = rgb_dir + 'oof_up560gaussfilt2down280/test/'
        gt_dir = rgb_dir + 'clean/test/'
        input_dir = rgb_dir + 'blurry/test/'
        print('val dataloader')
        print('using gt_dir: ', gt_dir)
        print('using input_dir: ', input_dir)

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        # self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        # self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_tiff_file(x) or is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_tiff_file(x) or is_png_file(x)]

        self.noisy_filenames = reorder_filenames(self.noisy_filenames, reference=self.clean_filenames,
                                                 token=self.img_options['token'])

        # # new test
        # root_folder = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/'
        # self.clean_filenames = [
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-31-18_351_434_138.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-28-108_14_282_138.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_5_w3soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-30-362_50_137_337.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_5_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-36-93_386_336_121.tif',
        #     root_folder + 'clean/210831_MultiviewTreeFrog_5_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-29-123_23_336_142.tif',
        # ]
        # self.noisy_filenames = [
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-31-18_351_434_138.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-28-108_14_282_138.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_5_w2soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-30-362_50_137_337.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_5_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-36-93_386_336_121.tif',
        #     root_folder + 'blurry/210831_MultiviewTreeFrog_5_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-29-123_23_336_142.tif',
        # ]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        # clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # clean = torch.from_numpy(np.float32(load_img_png(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img_png(self.noisy_filenames[tar_index])))

        # hack to train always with the same image

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/patches_128/5/210831_MultiviewTreeFrog_5_w2soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512_1239_uint8.png'
        # img_blurred = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/patches_128/5/210831_MultiviewTreeFrog_5_w3soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512_1239_uint8.png'
        # clean = torch.from_numpy(np.float32(load_img_png(img_clean)))
        # noisy = torch.from_numpy(np.float32(load_img_png(img_blurred)))

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif'
        # img_blurry = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-104_0_317_150.tif'

        # img_clean = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w5soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif'
        # img_blurry = '/home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected/210831_MultiviewTreeFrog_2_w4soSPIM-405_cropped_110_110_1024_1024_denoised_N2V_centercropped_down512-21-14_340_496_158.tif'
        #
        # clean = np.float32(load_img_tiff(img_clean))
        # noisy = np.float32(load_img_tiff(img_blurry))

        # clean = np.float32(load_img_tiff(self.clean_filenames[tar_index]))
        # noisy = np.float32(load_img_tiff(self.noisy_filenames[tar_index]))
        clean = np.float32(load_img_png(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img_png(self.noisy_filenames[tar_index]))

        n_channels = 1
        # clean = np.repeat(clean[..., np.newaxis], axis=2, repeats=n_channels)
        # noisy = np.repeat(noisy[..., np.newaxis], axis=2, repeats=n_channels)
        clean_ = np.repeat(clean[..., np.newaxis], axis=2, repeats=n_channels)
        noisy_ = np.repeat(noisy[..., np.newaxis], axis=2, repeats=n_channels)
        clean = np.zeros((512, 512, n_channels), dtype=np.float32)
        noisy = np.zeros((512, 512, n_channels), dtype=np.float32)
        clean[:clean_.shape[0], :clean_.shape[1]] = clean_
        noisy[:noisy_.shape[0], :noisy_.shape[1]] = noisy_

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        # if 'Mouse_actin' in self.root_folder:
        #     clean = clean[:512, :512]
        #     noisy = noisy[:512, :512]
        # if any(dn in self.root_folder for dn in ['Mouse_skull_nuclei', 'Multiview']):
        #     clean = clean[:256, :256]
        #     noisy = noisy[:256, :256]

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

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
