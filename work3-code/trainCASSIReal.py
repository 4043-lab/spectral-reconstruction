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
# if opt.train_dataset == "CAVE1024":
#     train_set = LoadTraining(opt.data_path)
# elif opt.train_dataset == "CAVE512":
#     train_set = LoadTrainingOnCAVE_512_28(opt.data_path)
# elif opt.train_dataset == "ICVL":
#     train_set = LoadTrainingOnICVL(opt.data_path)

# if opt.train_dataset.find("CAVE") >= 0:
#     test_data = LoadTestOnKAIST(opt.test_path)
# elif opt.train_dataset.find("ICVL") >= 0:
#     test_data = LoadTestOnICVL(opt.test_path)
# else:
#     test_data = None

train_set_KAIST = prepare_data_KAIST(opt.data_path_KAIST, 1)
train_set_CAVE1024 = prepare_data_cave(opt.data_path_CAVE, 1)
train_set = train_set_KAIST + train_set_CAVE1024

test_data = prepare_data(opt.data_path_real, 5)

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path_real, opt.input_mask, opt.batch_size, opt.in_channels, patch_size=opt.patch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path_real, opt.input_mask, 1, opt.in_channels, patch_size=opt.patch_size)

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

if opt.method == 'my_model':
    DAUHST_model = model_generator(opt, 'dauhst',
                               opt.pretrained_DAUHST_model_path).cuda()
    DAUHST_model = DAUHST_model.eval()

    # model.diffusion_prior.load_state_dict(
    #     torch.load("experiments/cassi_dauhst3stg_spa_l2loss_alltrainable_ddpm_240727_120756/checkpoint/I80000_E80_gen.pth"),
    #     strict=False)
    model.diffusion_prior.load_state_dict(
        torch.load(opt.pretrained_DDPM_model_path),
        strict=False)
    for k, v in model.diffusion_prior.named_parameters():
        v.requires_grad = False

    diffusion_param_ids = [id(p) for p in model.diffusion_prior.parameters()]
    deep_prior_params = [p for p in model.parameters() if id(p) not in diffusion_param_ids]

    optimizer = torch.optim.AdamW([{'params': deep_prior_params, 'lr': opt.learning_rate}], betas=(0.9, 0.999))
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

print('is Finetune:{}'.format(opt.isFinetune))
# optimizing
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate, betas=(0.9, 0.999))

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
        gt_batch = shuffle_crop_real(train_set, opt.batch_size, crop_size=opt.patch_size, channels=opt.in_channels, argument=True)

        input_meas, meas = init_meas_KAIST(gt_batch, mask3d_batch_train, opt.input_setting, opt.in_channels)

        label = torch.from_numpy(gt_batch).cuda().float()

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
            # meas_out, _ = init_meas_KAIST(model_out, mask3d_batch_train, opt.input_setting)
            loss = torch.sqrt(mse(model_out, label))# + torch.sqrt(mse(input_meas, meas_out))
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
    model.eval()
    begin = time.time()
    res = np.zeros((5, 660, 660, 28))
    for i in range(5):
        meas = test_data[:, :, i]
        meas = meas * 1.2 * 2
        meas = torch.FloatTensor(meas)
        meas = meas.unsqueeze(0)
        meas = Variable(meas)
        meas = meas.cuda()
        input_meas, meas = init_meas_real_test(meas, mask3d_batch_test, opt.input_setting, opt.in_channels)
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
            result = model_out
        res[i, :, :, :] = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
    end = time.time()

    pred = res

    logger.info('===> Epoch {}: testing time: {:.2f}'.format(epoch, (end - begin)))
    model.train()
    return pred

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}, epoch_sam_num:{}, max_epoch:{}, train_dataset:{}.\n".format(opt.learning_rate, opt.batch_size, opt.epoch_sam_num, opt.max_epoch, opt.train_dataset))
    logger.info("DAUHST path:{}.\n".format(opt.pretrained_DAUHST_model_path))
    logger.info("DDPM path:{}.\n".format(opt.pretrained_DDPM_model_path))
    for epoch in range(1, opt.max_epoch + 1):
        train_loss = train(epoch, logger)
        loss_list.append(str(epoch) + '\t' + str(train_loss) + '\n')
        pred = test(epoch, logger)
        scheduler.step()
        name = result_path + '/' + 'Test-{}.mat'.format(epoch)
        scio.savemat(name, {'pred': pred})
        checkpoint(model, epoch, model_path, logger)

    loss_save_path = model_path + 'loss.txt'

    with open(loss_save_path, "w") as fout:
        for i in loss_list:
            fout.write(i)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


