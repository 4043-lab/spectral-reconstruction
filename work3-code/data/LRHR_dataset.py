from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        if split == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = 'train_B'
            # gt_dir = 'Normal'
            # input_dir = 'Low'
        else:
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = 'test_B'
            # gt_dir = 'Normal'
            # input_dir = 'Low'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]
            # self.sr_path = Util.get_paths_from_images(
            #     '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # self.hr_path = Util.get_paths_from_images(
            #     '{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #     self.lr_path = Util.get_paths_from_images(
            #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:

            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.split == 'train':
                hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
            else:
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
            hr_name = hr_name.replace('_A', '_C')
            img_HR = Image.open(hr_name).convert("RGB")
            img_mask = Image.open(self.mask_path[index]).convert("1")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}
        else:
            [img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}

import scipy.io as sio
import numpy as np
import torch

def shift(inputs, step=2): # input [28,256,256]  output [28, 256, 310]
    [nC, row, col] = inputs.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i, :, step * i:step * i + col] = inputs[i, :, :]
    return output

def shift_back(inputs, step=2, in_channels=28):  # input [256,310]  output [28, 256, 256]
    [row, col] = inputs.shape
    # print(inputs.shape)
    nC = in_channels
    output = torch.zeros(nC, row, col - (nC - 1) * step)
    for i in range(nC):
        output[i, :, :] = inputs[:, step * i:step * i + col - (nC - 1) * step]
    return output

class CASSIDataSet(Dataset):
    def __init__(self, dataroot, r_resolution=128, data_len=-1, phase='train'):
        self.r_resolution = r_resolution
        self.data_len = data_len
        self.phase = phase
        if phase == 'train':
            # self.data_path = f"{dataroot}/CAVE_512_28/"
            # self.data_path = f"{dataroot}/cave_1024_28/"
            self.data_path = f"{dataroot}/ICVL/training_data/"
        else:
            # self.data_path = f"{dataroot}/TSA_simu_data/Truth/"
            self.data_path = f"{dataroot}/ICVL/testing_data/"
        self.mask_path = f"{dataroot}/TSA_simu_data/"
        self.scene_list = os.listdir(self.data_path)
        self.scene_list.sort()
        self.img_num = len(self.scene_list)
        self.dataset = self.LoadDataSet(self.data_path, phase)
        ## load mask
        data = sio.loadmat(self.mask_path + '/mask.mat')
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[np.newaxis, :, :], (24, 1, 1))

    def __len__(self):
        return self.data_len

    def LoadDataSet(self, path, phase='train'):
        imgs = []
        scene_list = os.listdir(path)
        scene_list.sort()
        if phase == 'train':
            print('training sences:', len(scene_list))
            # for i in range(len(scene_list)):
            #     # for i in range(5):
            #     scene_path = path + scene_list[i]
            #     scene_num = int(scene_list[i].split('.')[0][5:])
            #     if scene_num <= 30:
            #         if 'mat' not in scene_path:
            #             continue
            #         img_dict = sio.loadmat(scene_path)
            #         img = img_dict['data_slice']  # / 65536.
            #         img = img / np.max(img)
            #         img = img.astype(np.float32)
            #         imgs.append(img)
            #         print('Sence {} is loaded. {}'.format(i, scene_list[i]))

            #cave1024
            # for i in range(len(scene_list)):
            #     # for i in range(5):
            #     scene_path = path + scene_list[i]
            #     scene_num = int(scene_list[i].split('.')[0][5:])
            #     if scene_num<=205:
            #         img = None
            #         if 'mat' not in scene_path:
            #             continue
            #         img_dict = sio.loadmat(scene_path)
            #         if "img_expand" in img_dict:
            #             img = img_dict['img_expand'] / 65536.
            #             # img = img / np.max(img)
            #         elif "img" in img_dict:
            #             img = img_dict['img'] / 65536.
            #             # img = img / np.max(img)
            #         img = img.astype(np.float32)
            #         imgs.append(img)
            #         print('Sence {} is loaded. {}'.format(i, scene_list[i]))

            #ICVL
            imgs = []
            scene_list = os.listdir(path)
            scene_list.sort()
            print('training sences:', len(scene_list))
            for i in range(len(scene_list)):
            # for i in range(5):
                scene_path = path + scene_list[i]
                if 'mat' not in scene_path:
                    continue
                img_dict = sio.loadmat(scene_path)
                if "data" in img_dict:
                    img = img_dict['data']
                img = img.astype(np.float32)
                imgs.append(img)
                print('Sence {} is loaded. {}'.format(i, scene_list[i]))
            return imgs
        else:
            print('val sences:', len(scene_list))
            # for i in range(len(scene_list)):
            #     # for i in range(5):
            #     scene_path = path + scene_list[i]
            #     scene_num = int(scene_list[i].split('.')[0][5:])
            #     if scene_num <= 30:
            #         if 'mat' not in scene_path:
            #             continue
            #         img_dict = sio.loadmat(scene_path)
            #         img = img_dict['img']
            #         img = img.astype(np.float32)
            #         imgs.append(img)
            #         print('Sence {} is loaded. {}'.format(i, scene_list[i]))
            
            path_test = path + scene_list[0]
            test_data = sio.loadmat(path_test)['data']
            # test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
            return test_data
        
        # return imgs

    def arguement_1(self, x):
        """
        :param x: c,h,w
        :return: c,h,w
        """
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        # Random rotation
        for j in range(rotTimes):
            x = torch.rot90(x, dims=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            x = torch.flip(x, dims=(2,))
        # Random horizontal Flip
        for j in range(hFlip):
            x = torch.flip(x, dims=(1,))
        return x

    def arguement_2(self, generate_gt, whole_size=256):
        c, h, w = generate_gt.shape[1], whole_size, whole_size
        divid_point_h = h // 2
        divid_point_w = w // 2
        output_img = torch.zeros(c, h, w).cuda()
        output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
        output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
        output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
        output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
        return output_img

    def init_meas(self, gt, mask, input_setting='H', in_channels=28):
        input = None
        meas = None
        if input_setting == 'H':
            input, meas = self.gen_meas(gt, mask, Y2H=True, mul_mask=False, in_channels=in_channels)
        elif input_setting == 'HM':
            input, meas = self.gen_meas(gt, mask, Y2H=True, mul_mask=True, in_channels=in_channels)
        elif input_setting == 'Y':
            meas = self.gen_meas(gt, mask, Y2H=False, mul_mask=False, in_channels=in_channels)
        elif input_setting == 'X':
            input = gt
        return input, meas

    def gen_meas(self, data_batch, mask3d_batch, Y2H=True, mul_mask=False, in_channels=28):
        nC = data_batch.shape[0]
        temp = shift(mask3d_batch * data_batch, 2)
        meas = torch.sum(temp, 0)
        if Y2H:
            meas_norm = meas / nC * 2
            H = shift_back(meas_norm, step=2, in_channels=nC)
            if mul_mask:
                HM = torch.mul(H, mask3d_batch)
                return HM, meas
            return H, meas

        return meas

    def __getitem__(self, index):
        if self.phase == 'train':
            # choseAgument = random.randint(0, 2)
            # if(choseAgument % 3 == 1):
            #     sample_list = np.random.randint(0, self.img_num, 4)
            #     processed_data = np.zeros((4, self.r_resolution // 2, self.r_resolution // 2, 28), dtype=np.float32)
            #     for j in range(4):
            #         img = self.dataset[sample_list[j]]
            #         h, w, c = img.shape
            #         x_index = np.random.randint(0, h - self.r_resolution // 2)
            #         y_index = np.random.randint(0, w - self.r_resolution // 2)
            #         processed_data[j] = img[x_index:x_index + self.r_resolution // 2, y_index:y_index + self.r_resolution // 2, :]
            #     gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            #     label = self.arguement_1(self.arguement_2(gt_batch, self.r_resolution))
            # else:
            #     idx = random.randint(0, self.img_num-1)
            #     # processed_data = np.zeros((self.size, self.size, self.in_channels), dtype=np.float32)
            #     img = self.dataset[idx]
            #     h, w, c = img.shape
            #     x_index = np.random.randint(0, h - self.r_resolution)
            #     y_index = np.random.randint(0, w - self.r_resolution)
            #     processed_data = img[x_index:x_index + self.r_resolution, y_index:y_index + self.r_resolution, :]
            #     processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).cuda().float()
            #     label = self.arguement_1(processed_data)
            idx = random.randint(0, self.img_num - 1)
            # processed_data = np.zeros((self.size, self.size, self.in_channels), dtype=np.float32)
            img = self.dataset[idx]
            h, w, c = img.shape
            x_index = np.random.randint(0, h - self.r_resolution)
            y_index = np.random.randint(0, w - self.r_resolution)
            processed_data = img[x_index:x_index + self.r_resolution, y_index:y_index + self.r_resolution, :]
            processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).cuda().float()
            label = self.arguement_1(processed_data)
        else:
            processed_data = self.dataset[index-1, :, :, :]
            processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).cuda().float()
            label = processed_data

        pxm = random.randint(0, 256 - self.r_resolution)
        pym = random.randint(0, 256 - self.r_resolution)
        mask_3d = self.mask_3d[:, pxm:pxm + self.r_resolution:1, pym:pym + self.r_resolution:1]
        mask_3d_torch = torch.from_numpy(mask_3d).cuda().float()

        SR_img, meas = self.init_meas(label, mask_3d_torch)
        Phi = shift(mask_3d_torch, 2)

        Phi_s = torch.sum(Phi ** 2, dim=0, keepdim=False)  # (H, W+(ch-1)*2)
        Phi_s[Phi_s == 0] = 1

        return {'HR': label, 'SR': SR_img, 'mask': mask_3d_torch[0:1, :, :], 'mask_3d': mask_3d_torch, 'Index': index, 'meas': meas, 'Phi': Phi, 'Phi_s': Phi_s}




