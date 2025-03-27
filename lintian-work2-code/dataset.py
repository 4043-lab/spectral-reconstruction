import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import os
import logging
import sys


class dataset_for_training(tud.Dataset):
    def __init__(self, opt, train_set):
        super(dataset_for_training, self).__init__()
        self.size = opt.patch_size
        self.trainset_num = opt.epoch_sam_num
        self.data_path = opt.data_path
        self.scene_list = os.listdir(self.data_path)
        self.scene_list.sort()
        self.img_num = len(self.scene_list)
        self.train_set = train_set
        self.in_channels = opt.in_channels
        self.train_dataset = opt.train_dataset
        self.method = opt.method

        ## load mask
        data = sio.loadmat(opt.mask_path + '/mask.mat')
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[np.newaxis, :, :], (self.in_channels, 1, 1))

    def __getitem__(self, index):
        if self.method.find("my_model") >= 0:
            index1 = random.randint(0, self.img_num)
            # processed_data = np.zeros((self.size, self.size, self.in_channels), dtype=np.float32)
            img = self.train_set[index1]
            h, w, c = img.shape
            assert c == self.in_channels
            x_index = np.random.randint(0, h - self.size)
            y_index = np.random.randint(0, w - self.size)
            processed_data = img[x_index:x_index + self.size, y_index:y_index + self.size, :]
            processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).cuda().float()
            label = self.arguement_1(processed_data)
            # sample_list = np.random.randint(0, self.img_num, 4)
            # processed_data = np.zeros((4, self.size // 2, self.size // 2, self.in_channels), dtype=np.float32)
            # for j in range(4):
            #     img = self.train_set[sample_list[j]]
            #     h, w, c = img.shape
            #     assert c == self.in_channels
            #     x_index = np.random.randint(0, h - self.size // 2)
            #     y_index = np.random.randint(0, w - self.size // 2)
            #     processed_data[j] = img[x_index:x_index + self.size // 2, y_index:y_index + self.size // 2, :]
            # gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            # label = self.arguement_1(self.arguement_2(gt_batch, self.size))
        else:
            index1 = random.randint(0, self.img_num)
            hsi = self.train_set[index1]
            shape = np.shape(hsi)

            px = random.randint(0, shape[0] - self.size)
            py = random.randint(0, shape[1] - self.size)
            label = hsi[px:px + self.size:1, py:py + self.size:1, :]

            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label = np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

            label = torch.FloatTensor(label.copy()).permute(2, 0, 1)

        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[:, pxm:pxm + self.size:1, pym:pym + self.size:1]

        # mask_3d_shift = np.zeros((self.in_channels, self.size, self.size + (self.in_channels - 1) * 2))
        # mask_3d_shift[:, :, 0:self.size] = mask_3d
        # for t in range(self.in_channels):
        #     mask_3d_shift[t, :, :] = np.roll(mask_3d_shift[t, :, :], 2 * t, axis=1)
        # mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=0, keepdims=False)
        # mask_3d_shift_s[mask_3d_shift_s == 0] = 1

        # mask_3d = torch.FloatTensor(mask_3d.copy()).permute(2, 0, 1)
        # Phi = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        # Phi_s = torch.FloatTensor(mask_3d_shift_s.copy())
        
        return label, mask_3d

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

    def __len__(self):
        return self.trainset_num
