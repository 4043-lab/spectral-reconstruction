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


if opt.train_dataset.find("CAVE") >= 0:
    test_data = LoadTestOnKAIST(opt.test_path)
elif opt.train_dataset.find("ICVL") >= 0:
    test_data = LoadTestOnICVL(opt.test_path)
else:
    test_data = None

# init mask
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10, opt.in_channels)

result_path = opt.outf + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model
model = model_generator(opt, opt.method, opt.pretrained_model_path, diffusion_opt)
model = dataparallel(model, opt)

DAUHST_model = model_generator(opt, 'dauhst',
                               opt.pretrained_DAUHST_model_path).cuda()
DAUHST_model = DAUHST_model.eval()

model.diffusion_prior.load_state_dict(
    torch.load(opt.pretrained_DDPM_model_path),
    strict=False)
for k, v in model.diffusion_prior.named_parameters():
    v.requires_grad = False

def test():
    psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all = [], [], [], [], [], []
    label = test_data.cuda().float()
    input_meas, meas = init_meas(label, mask3d_batch_test, opt.input_setting, opt.in_channels)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method == 'specat':
            model_out = model(input_meas, input_mask_test)
        elif opt.method == 'dauhst':
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
            model_out = model(input_meas, input_mask_test)

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

    print('===>: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, rmse = {:.3f}, ergas = {:.3f}, scc = {:.3f}, time: {:.2f}'
                .format(psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean, (end - begin)))
    return pred, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    (pred, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean) = test()
    name = result_path + '/' + 'Test-test.mat'
    scio.savemat(name, {'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all, 'sam_list': sam_all, 'rmse_list': rmse_all, 'ergas_list': ergas_all, 'scc_list': scc_all})



