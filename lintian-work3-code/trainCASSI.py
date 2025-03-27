from architecture import *
from cassi_utils import *
import torch
import torch.nn as nn
import scipy.io as scio
import time
import torch.utils.data as tud
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt, diffusion_opt
import torch.nn.functional as F
import data as Data
import model.networks as networks

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# load dataset
if opt.train_dataset == "CAVE1024":
    train_set = LoadTraining(opt.data_path)
elif opt.train_dataset == "CAVE512":
    train_set = LoadTrainingOnCAVE_512_28(opt.data_path)
elif opt.train_dataset == "ICVL":
    train_set = LoadTrainingOnICVL(opt.data_path)

if opt.train_dataset.find("CAVE") >= 0:
    test_data = LoadTestOnKAIST(opt.test_path)
elif opt.train_dataset.find("ICVL") >= 0:
    test_data = LoadTestOnICVL(opt.test_path)
else:
    test_data = None

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size, opt.in_channels)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10, opt.in_channels)

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
model = model_generator(opt, opt.method, opt.pretrained_model_path, diffusion_opt)
model = dataparallel(model, opt)

# DAUHST_model = model_generator(opt, 'dauhst',
#                                opt.pretrained_DAUHST_model_path).cuda()
# DAUHST_model = DAUHST_model.eval()

# # model.diffusion_prior.load_state_dict(
# #     torch.load("experiments/cassi_dauhst3stg_spa_l2loss_alltrainable_ddpm_240727_120756/checkpoint/I80000_E80_gen.pth"),
# #     strict=False)
# model.diffusion_prior.load_state_dict(
#     torch.load(opt.pretrained_DDPM_model_path),
#     strict=False)
# for k, v in model.diffusion_prior.named_parameters():
#     v.requires_grad = False

# diffusion_param_ids = [id(p) for p in model.diffusion_prior.parameters()]
# deep_prior_params = [p for p in model.parameters() if id(p) not in diffusion_param_ids]

# optimizer = torch.optim.AdamW([{'params': deep_prior_params, 'lr': opt.learning_rate}], betas=(0.9, 0.999))

# print('is Finetune:{}'.format(opt.isFinetune))

# optimizing
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate, betas=(0.9, 0.999))

if opt.scheduler == 'MultiStepLR' and not opt.isFinetune:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR' and not opt.isFinetune:
    if opt.method == 'my_model':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=5e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=5e-6)
mse = torch.nn.MSELoss().cuda()
# criterion = nn.L1Loss().cuda()

loss_list = []
psnr_list = []
ssim_list = []
sam_list = []
rmse_list = []
ergas_list = []
scc_list = []

# for phase, dataset_opt in diffusion_opt['datasets'].items():
#     if phase == 'train':
#         train_set = Data.create_dataset(dataset_opt, phase)
#         train_loader = Data.create_dataloader(
#             train_set, dataset_opt, phase)
#     elif phase == 'val':
#         val_set = Data.create_dataset(dataset_opt, phase)
#         val_loader = Data.create_dataloader(
#             val_set, dataset_opt, phase)

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    iters = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for i in range(iters):
        print("epoch:{}/iteration:{}".format(epoch, i))
        gt_batch = shuffle_crop_my(train_set, opt.batch_size, crop_size=opt.patch_size, channels=opt.in_channels, argument=True)
        label = Variable(gt_batch).cuda().float()

        input_meas, meas = init_meas(label, mask3d_batch_train, opt.input_setting, opt.in_channels)

        optimizer.zero_grad()

        if opt.method == 'specat':
            model_out = model(input_meas, input_mask_train)
        elif opt.method == 'dauhst' or opt.method == 'dhm':
            Phi, Phi_s = input_mask_train
            model_out = model(input_meas, Phi, Phi_s)
        elif opt.method == 'my_model' or opt.method == 'my_model_onlyDiff':
            Phi, Phi_s = input_mask_train
            with torch.no_grad():
                cond = DAUHST_model(meas, Phi, Phi_s)
            model_out, v = model(meas, input_meas, mask3d_batch_train, Phi, Phi_s, cond)
        elif opt.method == 'my_model_onlyMamba':
            Phi, Phi_s = input_mask_train
            model_out, v = model(meas, input_meas, mask3d_batch_train, Phi, Phi_s)
        else:
            model_out = model(input_meas, input_mask_train)

        if opt.method == 'specat':
            meas_out, _ = init_meas(model_out, mask3d_batch_train, opt.input_setting)
            loss = torch.sqrt(mse(model_out, label)) + torch.sqrt(mse(input_meas, meas_out))
        elif opt.method == 'my_model':
            # meas_out = shift(v * label).cuda()
            # meas_out = torch.sum(meas_out, dim=1, keepdim=True)
            # meas_gt = meas.unsqueeze(1)
            loss = mse(model_out, label)# + 0.001 * mse(meas_gt, meas_out)
        else:
            loss = mse(model_out, label) # + 0.5 * mse(model_out[-1], label) + 0.2 * mse(model_out[-2], label) + 0.1 * mse(meas_out, meas_gt)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / iters, (end - begin)))
    return epoch_loss / iters

def test(epoch, logger):
    psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all = [], [], [], [], [], []
    label = test_data.cuda().float()
    input_meas, meas = init_meas(label, mask3d_batch_test, opt.input_setting, opt.in_channels)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method == 'specat':
            model_out = model(input_meas, input_mask_test)
        elif opt.method == 'dauhst' or opt.method == 'dhm':
            Phi, Phi_s = input_mask_test
            model_out = model(input_meas, Phi, Phi_s)
        elif opt.method == 'my_model' or opt.method == 'my_model_onlyDiff':
            Phi, Phi_s = input_mask_test
            cond = DAUHST_model(meas, Phi, Phi_s)
            model_out, v = model(meas, input_meas, mask3d_batch_test, Phi, Phi_s, cond)
        elif opt.method == 'my_model_onlyMamba':
            Phi, Phi_s = input_mask_test
            model_out, v = model(meas, input_meas, mask3d_batch_test, Phi, Phi_s)
        else:
            model_out = model(input_meas, input_mask_train)

    for k in range(label.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], label[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], label[k, :, :, :])
        sam_val = torch_sam(model_out[k, :, :, :], label[k, :, :, :])
        rmse_val = torch_rmse(model_out[k, :, :, :], label[k, :, :, :])
        ergas_val = torch_ergas(model_out[k, :, :, :], label[k, :, :, :])
        scc_val = torch_scc(model_out[k, :, :, :], label[k, :, :, :])

        psnr_all.append(psnr_val.detach().cpu().numpy())
        ssim_all.append(ssim_val.detach().cpu().numpy())
        sam_all.append(sam_val.detach().cpu().numpy())
        rmse_all.append(rmse_val)
        ergas_all.append(ergas_val)
        scc_all.append(scc_val)

    end = time.time()

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    # truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_all))
    ssim_mean = np.mean(np.asarray(ssim_all))
    sam_mean = np.mean(np.asarray(sam_all))
    rmse_mean = np.mean(np.asarray(rmse_all))
    ergas_mean = np.mean(np.asarray(ergas_all))
    scc_mean = np.mean(np.asarray(scc_all))

    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, rmse = {:.3f}, ergas = {:.3f}, scc = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean, (end - begin)))
    model.train()
    return pred, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}, epoch_sam_num:{}, max_epoch:{}, train_dataset:{}.\n".format(opt.learning_rate, opt.batch_size, opt.epoch_sam_num, opt.max_epoch, opt.train_dataset))
    logger.info("DAUHST path:{}.\n".format(opt.pretrained_DAUHST_model_path))
    logger.info("DDPM path:{}.\n".format(opt.pretrained_DDPM_model_path))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train_loss = train(epoch, logger)
        loss_list.append(str(epoch) + '\t' + str(train_loss) + '\n')
        (pred, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean) = test(epoch, logger)
        psnr_list.append(str(epoch) + '\t' + str(psnr_mean) + '\n')
        ssim_list.append(str(epoch) + '\t' + str(ssim_mean) + '\n')
        sam_list.append(str(epoch) + '\t' + str(sam_mean) + '\n')
        rmse_list.append(str(epoch) + '\t' + str(rmse_mean) + '\n')
        ergas_list.append(str(epoch) + '\t' + str(ergas_mean) + '\n')
        scc_list.append(str(epoch) + '\t' + str(scc_mean) + '\n')
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 20:
                name = result_path + '/' + 'Test.mat'
                scio.savemat(name, {'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all, 'sam_list': sam_all, 'rmse_list': rmse_all, 'ergas_list': ergas_all, 'scc_list': scc_all})
                checkpoint(model, epoch, model_path, logger, psnr_mean, ssim_mean)

    loss_save_path = model_path + 'loss.txt'
    psnr_save_path = model_path + 'psnr.txt'
    ssim_save_path = model_path + 'ssim.txt'
    sam_save_path = model_path + 'sam.txt'
    rmse_save_path = model_path + 'rmse.txt'
    ergas_save_path = model_path + 'ergas.txt'
    scc_save_path = model_path + 'scc.txt'

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
    with open(rmse_save_path, "w") as fout:
        for i in rmse_list:
            fout.write(i)
    with open(ergas_save_path, "w") as fout:
        for i in ergas_list:
            fout.write(i)
    with open(scc_save_path, "w") as fout:
        for i in scc_list:
            fout.write(i)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


