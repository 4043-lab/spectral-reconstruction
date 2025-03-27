from architecture import *
from utils import *
import torch
import torch.nn as nn
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)
phi = shift(mask3d_batch_train)
# dataset
if opt.train_dataset=="CAVE1024":
    train_set = LoadTraining(opt.data_path)
elif opt.train_dataset=="CAVE512":
    train_set = LoadTrainingOnCAVE_512_28(opt.data_path)
test_data = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
    model=dataparallel(model, opt)
    FDL_loss=FDL_loss.cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path)
    model = dataparallel(model, opt)

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=5e-5)
mse = torch.nn.MSELoss().cuda()
criterion = nn.L1Loss().cuda()
loss_list = []
psnr_list = []
ssim_list = []
sam_list = []

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for i in range(batch_num):
        print("epoch:{}/iteration:{}".format(epoch,i))
        if opt.method.find('my_model') >= 0:
            gt_batch = shuffle_crop_my(train_set, opt.batch_size, crop_size=opt.patch_size, channels=28, argument=True)
        else:
            gt_batch = shuffle_crop(train_set, opt.batch_size, crop_size=opt.patch_size, channels=28, argument=True)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
        optimizer.zero_grad()
        # meas_gt = shift(mask3d_batch_train * gt).cuda()
        # meas_gt = torch.sum(meas_gt, dim=1, keepdim=True)

        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, diff_pred = model(input_meas, input_mask_train)
            loss = torch.sqrt(mse(model_out, gt))
            diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
            loss_sparsity = F.mse_loss(diff_gt, diff_pred)
            loss = loss + 2 * loss_sparsity
        elif opt.method == 'herosnet':
            model_out, _ = model(input_meas, input_mask_train)
            loss = criterion(model_out[7], gt) + 0.5 * criterion(model_out[6], gt) + 0.5 * criterion(model_out[5], gt)
        elif opt.method.find('my_model')>=0:
            model_out, optim_mask = model(input_meas, input_mask_train)
            meas_out = shift(optim_mask[0] * gt).cuda()
            meas_out = torch.sum(meas_out, dim=1, keepdim=True)
            meas_gt = input_meas.unsqueeze(1)
            loss = mse(model_out, gt) + 0.001 * mse(meas_out, meas_gt) + 0.0005 * mse(optim_mask[2], meas_gt)# + 0.5 * mse(model_out[-2], gt) + 0.5 * mse(model_out[-3], gt)
        elif opt.method == 'dgsmp':
            model_out = model(input_meas, input_mask_train)
            loss = criterion(model_out, gt)
        else:
            model_out = model(input_meas, input_mask_train)
            loss = torch.sqrt(mse(model_out, gt))

        if opt.method == 'hdnet':
            fdl_loss = FDL_loss(model_out, gt)
            loss = loss + 0.7 * fdl_loss

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    return epoch_loss / batch_num

def test(epoch, logger):
    psnr_all, ssim_all, sam_all = [], [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_meas, input_mask_test)
        elif opt.method.find('my_model')>=0:
            model_out, _ = model(input_meas, input_mask_test)
            # model_out = out[-1]
        elif opt.method == 'herosnet':
            out, _ = model(input_meas, input_mask_train)
            model_out = out[7]
        else:
            model_out = model(input_meas, input_mask_test)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        sam_val = torch_sam(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_all.append(psnr_val.detach().cpu().numpy())
        ssim_all.append(ssim_val.detach().cpu().numpy())
        sam_all.append(sam_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_all))
    ssim_mean = np.mean(np.asarray(ssim_all))
    sam_mean = np.mean(np.asarray(sam_all))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean, sam_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train_loss=train(epoch, logger)
        loss_list.append(str(epoch) + '\t' + str(train_loss) + '\n')
        (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = test(epoch, logger)
        psnr_list.append(str(epoch) + '\t' + str(psnr_mean) + '\n')
        ssim_list.append(str(epoch) + '\t' + str(ssim_mean) + '\n')
        sam_list.append(str(epoch) + '\t' + str(sam_mean) + '\n')
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 20:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}_{:.3f}'.format(epoch, psnr_max, ssim_mean, sam_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all, 'sam_list': sam_all})
                checkpoint(model, epoch, model_path, logger)

    loss_save_path = model_path + 'loss.txt'
    psnr_save_path = model_path + 'psnr.txt'
    ssim_save_path = model_path + 'ssim.txt'
    sam_save_path = model_path + 'sam.txt'

    with open(loss_save_path, "w") as fout:
        for i in loss_list:
            fout.write(i)
    with open(psnr_save_path, "w") as fout:
        for i in psnr_list:
            fout.write(i)
    with open(ssim_save_path, "w") as fout:
        for i in ssim_list:
            fout.write(i)
    with open(sam_save_path, "w") as fout:
        for i in sam_list:
            fout.write(i)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


