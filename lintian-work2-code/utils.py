import sys
import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim

def generate_masks(mask_path, batch_size, inchannels=28):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, inchannels))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_random_masks(mask_path, batch_size, inchannels=28):
    mask = sio.loadmat(mask_path + '/random_mask.mat')
    mask = mask['random_mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, inchannels))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_shift_masks(mask_path, batch_size, inchannels=28):
    # mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    # mask_3d_shift = mask['mask_3d_shift']
    # mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    # mask_3d_shift = torch.from_numpy(mask_3d_shift)
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, inchannels))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()

    # [nC, H, W] = mask_3d_shift.shape
    Phi_batch = shift(mask3d_batch)#mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)#(batch_size, H, W)
    Phi_s_batch[Phi_s_batch==0] = 1

    return Phi_batch, Phi_s_batch

def shift_masks(mask3d_batch):
    Phi_batch = shift(mask3d_batch)#mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)#(batch_size, H, W)
    Phi_s_batch[Phi_s_batch==0] = 1

    return Phi_batch, Phi_s_batch

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
    # for i in range(5):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            img = None
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand']# / 65536.
                img = img / np.max(img)
            elif "img" in img_dict:
                img = img_dict['img']# / 65536.
                img = img / np.max(img)
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTrainingOnCAVE_512_28(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
    # for i in range(5):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=30:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "data_slice" in img_dict:
                img = img_dict['data_slice']# / 65536.
                img = img / np.max(img)
            else:
                logging.error("training data load failed")
                sys.exit(0)
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTrainingOnICVL(path):
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
        else:
            logging.error("training data load failed")
            sys.exit(0)
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTestOnKAIST(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadTestOnICVL(path_test):
    scene_list = os.listdir(path_test)
    path_test = path_test+scene_list[0]
    test_data = sio.loadmat(path_test)['data']
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

def torch_sam(img, ref):# input [C,H,W]
    """SAM for 3D image, shape (C, H, W); uint or float[0, 1]"""
    img1_ = img
    img2_ = ref
    inner_product = (img1_ * img2_).sum(axis=0)
    img1_spectral_norm = torch.sqrt((img1_**2).sum(axis=0))
    img2_spectral_norm = torch.sqrt((img2_**2).sum(axis=0))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm+0.0000000000001)).clip(min=-1, max=1)
    return torch.mean(torch.arccos(cos_theta))

def sam(img1, ref):
    """SAM for 3D image, shape (C, H, W); uint or float[0, 1]"""
    img1_ = img1
    img2_ = ref
    inner_product = (img1_ * img2_).sum(axis=0)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=0))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=0))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm+0.0000000000001)).clip(min=-1, max=1)
    return np.mean(np.arccos(cos_theta))

def torch_psnr(img, ref):  # input [C,H,W]
    img = img*255
    ref = ref*255
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def psnr(img, ref):  # input [C,H,W]
    img = img*255
    ref = ref*255
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * np.log10((255*255)/mse)
    return psnr / nC

def rmse(fake_hr, real_hr):
    channels = real_hr.shape[0]
    fake_hr = fake_hr.astype(np.float64).transpose((1, 2, 0))#(C,H,W) to (H,W,C)
    real_hr = real_hr.astype(np.float64).transpose((1, 2, 0))#(C,H,W) to (H,W,C)

    def single_rmse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return np.sqrt(mse)

    rmse_sum = 0
    for band in range(channels):
        fake_band_img = fake_hr[:, :, band]
        real_band_img = real_hr[:, :, band]
        rmse_sum += single_rmse(fake_band_img, real_band_img)

    rmse = rmse_sum / channels

    return rmse

def torch_rmse(fake_hr, real_hr):
    channels = real_hr.shape[0]
    fake_hr = fake_hr.detach().cpu().numpy().astype(np.float64).transpose((1, 2, 0))#(C,H,W) to (H,W,C)
    real_hr = real_hr.detach().cpu().numpy().astype(np.float64).transpose((1, 2, 0))#(C,H,W) to (H,W,C)

    def single_rmse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return np.sqrt(mse)

    rmse_sum = 0
    for band in range(channels):
        fake_band_img = fake_hr[:, :, band]
        real_band_img = real_hr[:, :, band]
        rmse_sum += single_rmse(fake_band_img, real_band_img)

    rmse = rmse_sum / channels

    return rmse

def ergas(img_fake, img_real, scale=1):
    """ERGAS for 2D (H, W) or 3D (C, H, W) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 1."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + 0.0000000000001))
    elif img_fake_.ndim == 3:
        img_fake_ = img_fake_.transpose((1, 2, 0))  # (C,H,W) to (H,W,C)
        img_real_ = img_real_.transpose((1, 2, 0))  # (C,H,W) to (H,W,C)
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + 0.0000000000001)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def torch_ergas(img_fake, img_real, scale=1):
    """ERGAS for 2D (H, W) or 3D (C, H, W) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 1."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.detach().cpu().numpy().astype(np.float64)
    img_real_ = img_real.detach().cpu().numpy().astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + 0.0000000000001))
    elif img_fake_.ndim == 3:
        img_fake_ = img_fake_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        img_real_ = img_real_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + 0.0000000000001)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def torch_scc(img, ref):
    """SCC for 2D (H, W)or 3D (C, H, W) image; uint or float[0, 1]"""
    if not  img.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img.detach().cpu().numpy().astype(np.float64)
    img2_ = ref.detach().cpu().numpy().astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        img1_ = img1_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        img2_ = img2_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def scc(img, ref):
    """SCC for 2D (H, W)or 3D (C, H, W) image; uint or float[0, 1]"""
    if not  img.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img.astype(np.float64)
    img2_ = ref.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        img1_ = img1_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        img2_ = img2_.transpose((1, 2, 0))#(C,H,W) to (H,W,C)
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def torch_ssim(img, ref):  # input [C,H,W]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def numpy_ssim(img, ref):  # input [C,H,W]
    img = torch.DoubleTensor(img)
    ref = torch.DoubleTensor(ref)
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=256, channels=28, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, channels), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, c = img.shape
            assert c == channels
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, channels), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                h, w, c = train_data[sample_list[j]].shape
                assert c == channels
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2, y_index:y_index+crop_size//2, :]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2, crop_size))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def shuffle_crop_my(train_data, batch_size, crop_size=256, channels=28, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size % 2)
        processed_data = np.zeros((batch_size % 2, crop_size, crop_size, channels), dtype=np.float32)
        for i in range(batch_size % 2):
            img = train_data[index[i]]
            h, w, c = img.shape
            assert c == channels
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, crop_size//2, crop_size//2, channels), dtype=np.float32)
        for i in range(batch_size - batch_size % 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                h, w, c = train_data[sample_list[j]].shape
                assert c == channels
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2, y_index:y_index+crop_size//2, :]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_1(arguement_2(gt_batch_2, crop_size)))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def arguement_1(x):
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

def arguement_2(generate_gt, whole_size=256):
    c, h, w = generate_gt.shape[1], whole_size, whole_size
    divid_point_h = h//2
    divid_point_w = w//2
    output_img = torch.zeros(c, h, w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False, in_channels=28):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas_norm = meas / nC * 2
        H = shift_back(meas_norm, step=2, in_channels=in_channels)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H

    return meas

def gen_meas_torch_my(data_batch, mask3d_batch, Y2H=True, mul_mask=False, in_channels=28):
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    compressed_mask = shift(mask3d_batch, 2)
    compressed_mask = torch.sum(compressed_mask, dim=1)
    if Y2H:
        meas_norm = meas / compressed_mask
        H = shift_back(meas_norm, step=2, in_channels=in_channels)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return (meas, H)

    return meas

def shift(inputs, step=2): # input [bs,28,256,256]  output [bs, 28, 256, 310]
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2, in_channels=28):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    # print(inputs.shape)
    nC = in_channels
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def init_mask(mask_path, mask_type, batch_size, in_channels=28):
    input_mask = None
    mask3d_batch = generate_masks(mask_path, batch_size, in_channels)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size, in_channels)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == 'Random_Mask':
        input_mask = generate_random_masks(mask_path, batch_size, in_channels)
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_train_mask(mask3d_batch, mask_type):
    input_mask = None
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = shift_masks(mask3d_batch)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_meas(gt, mask, input_setting, in_channels=28):
    input_meas = None
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False, in_channels=in_channels)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True, in_channels=in_channels)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=False, in_channels=in_channels)
    elif input_setting == 'X':
        input_meas = gt
    return input_meas

def init_meas_my(gt, mask, input_setting, in_channels=28):
    input_meas = None
    if input_setting == 'H':
        input_meas = gen_meas_torch_my(gt, mask, Y2H=True, mul_mask=False, in_channels=in_channels)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch_my(gt, mask, Y2H=True, mul_mask=True, in_channels=in_channels)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch_my(gt, mask, Y2H=False, mul_mask=False, in_channels=in_channels)
    elif input_setting == 'X':
        input_meas = gt
    return input_meas

def checkpoint(model, epoch, model_path, logger, psnr, ssim):
    model_out_path = model_path + "/model_epoch_{}_{:.2f}_{:.3f}.pth".format(epoch, psnr, ssim)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def dataparallel(model, opt):
    # if ngpus==0:
    #     assert False, "only support gpu mode"
    gpu_list =[int(i) for i in opt.gpu_id.split(",")]
    print("gpu_list:{},type:{}".format(gpu_list, type(gpu_list[-1])))
    # gpu_list = list(range(gpu0, gpu0+ngpus))
    # assert torch.cuda.device_count() >= gpu0 + ngpus
    ngpus=len(gpu_list)
    gpu_list = list(range(0, ngpus))
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    else:
        logging.error("no GPU!")
        sys.exit(0)
    return model

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ptflops import get_model_complexity_info

def my_summary(test_model, H = 256, W = 256, C = 28, N = 1):
    model = test_model.cuda()
    # print(model)
    inputs = torch.randn((N, C, H, W)).cuda()
    mask = torch.randn((N, C, H, W)).cuda()
    # print(parameter_count_table(model))
    flops = FlopCountAnalysis(model, (inputs, mask))
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'FC, GMac:{flops.total()/1e9}')
    print(f'Params:{n_param/1e6}')

    flops, params = profile(model, inputs=(inputs,))
    print("thop, FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("thop, Params=", str(params / 1e6) + '{}'.format("M"))

    macs, params = get_model_complexity_info(model, (C, H, W), as_strings=True, print_per_layer_stat=False)
    print('ptflops, MACs:  ' + macs)
    print('ptflops, Params: ' + params)
