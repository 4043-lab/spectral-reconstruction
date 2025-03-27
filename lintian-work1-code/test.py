from architecture import *
from utils import *
import torch
import torch.nn as nn
import scipy.io as scio
import time
import os
import numpy as np
import logging
from option import opt
from architecture.My_model import *

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
# init mask
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)

# dataset
test_data = LoadTest(opt.test_path)

# saving path
result_path = opt.outf + '/test_result'# + opt.method
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
    model=dataparallel(model, opt)
    FDL_loss=FDL_loss.cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path)
    model = dataparallel(model, opt)

# loss_list = []
# psnr_list = []
# ssim_list = []

def test():
    psnr_list, ssim_list, sam_list, rmse_list, ergas_list, scc_list = [], [], [], [], [], []
    test_gt = test_data.cuda().float()
    print("test_gt shape:{}".format(test_gt.shape))
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_meas, input_mask_test)
        elif opt.method.find('my_model') >= 0:
            model_out, phi = model(input_meas, input_mask_test)
        elif opt.method == 'herosnet':
            out, _ = model(input_meas, input_mask_test)
            model_out = out[7]
        else:
            model_out = model(input_meas, input_mask_test)
    end = time.time()
    print("model_out shape:{}".format(model_out.shape))
    
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        sam_val = torch_sam(model_out[k, :, :, :], test_gt[k, :, :, :])
        rmse_val = torch_rmse(model_out[k, :, :, :], test_gt[k, :, :, :])
        ergas_val = torch_ergas(model_out[k, :, :, :], test_gt[k, :, :, :])
        scc_val = torch_scc(model_out[k, :, :, :], test_gt[k, :, :, :])

        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val.detach().cpu().numpy())
        rmse_list.append(rmse_val)
        ergas_list.append(ergas_val)
        scc_list.append(scc_val)

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))
    rmse_mean = np.mean(np.asarray(rmse_list))
    ergas_mean = np.mean(np.asarray(ergas_list))
    scc_mean = np.mean(np.asarray(scc_list))

    print(
        '===> testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, rmse = {:.3f}, ergas = {:.3f}, scc = {:.3f}, time: {:.2f}'.format(
            psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean, (end - begin)))
    print("PSNR: {}".format(psnr_list))
    print("SSIM: {}".format(ssim_list))
    print("SAM: {}".format(sam_list))
    print("RMSE: {}".format(rmse_list))
    print("ERGAS: {}".format(ergas_list))
    print("SCC: {}".format(scc_list))

    if opt.method.find('my_model') >= 0:
        mask = phi[0].cpu().numpy()
        phi_res = np.transpose(phi[1].cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        meas_pre = np.transpose(phi[2].cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        alphas = phi[3]
        betas = phi[4].cpu().numpy()
        return pred, truth, psnr_list, ssim_list, sam_list, rmse_list, ergas_list, scc_list, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean, phi_res, mask, meas_pre, alphas, betas
    return pred, truth, psnr_list, ssim_list, sam_list, rmse_list, ergas_list, scc_list, psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if opt.method.find('my_model') >= 0:
        (pred, truth, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean,
         rmse_mean, ergas_mean, scc_mean, phi_res, mask, meas_pre, alphas, betas) = test()
        name = result_path + '/' + 'my_model.mat'#'Test_psnr:{:.2f}_ssim:{:.3f}_sam:{:.3f}_rmse:{:.3f}_ergas:{:.3f}_scc:{:.3f}'.format(
            # psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean) + '.mat'
        scio.savemat(name, {'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all, 'sam_list': sam_all,
                            'rmse_list': rmse_all, 'ergas_list': ergas_all, 'scc_list': scc_all, 'phi_res': phi_res,
                            'mask': mask, 'meas_pre': meas_pre, 'alphas': alphas, 'betas': betas})
    else:
        (pred, truth, psnr_all, ssim_all, sam_all, rmse_all, ergas_all, scc_all, psnr_mean, ssim_mean, sam_mean,
         rmse_mean, ergas_mean, scc_mean) = test()
        name = result_path + '/' + 'Test_psnr:{:.2f}_ssim:{:.3f}_sam:{:.3f}_rmse:{:.3f}_ergas:{:.3f}_scc:{:.3f}'.format(
            psnr_mean, ssim_mean, sam_mean, rmse_mean, ergas_mean, scc_mean) + '.mat'
        scio.savemat(name,
                     {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all, 'sam_list': sam_all,
                      'rmse_list': rmse_all, 'ergas_list': ergas_all, 'scc_list': scc_all})
    print("complete!")

