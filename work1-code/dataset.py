import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio


class dataset(tud.Dataset):
    def __init__(self, opt, CAVE):
        super(dataset, self).__init__()
        self.size = opt.size
        # self.path = opt.data_path
        self.num = opt.trainset_num
        self.CAVE = CAVE


    def __getitem__(self, index):
        index1 = random.randint(0, 29)
        hsi = self.CAVE[:, :, :, index1]
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

        label = torch.FloatTensor(label.copy()).permute(2,0,1)

        return label

    def __len__(self):
        return self.num
