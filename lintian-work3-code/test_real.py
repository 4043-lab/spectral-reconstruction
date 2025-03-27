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

test_data = prepare_data(opt.data_path_real, 5)

# init mask
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path_real, opt.input_mask, 1, opt.in_channels, patch_size=opt.patch_size)

# saving path
result_path = opt.outf
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model
model = model_generator(opt, opt.method, opt.pretrained_model_path, diffusion_opt)
model = dataparallel(model, opt)

print('is Finetune:{}'.format(opt.isFinetune))


def test():
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
            result = model_out
        res[i, :, :, :] = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
    end = time.time()

    pred = res

    print('===>: testing time: {:.2f}'.format((end - begin)))

    return pred

def main():
    pred = test()
    name = result_path + '/' + 'Test.mat'
    scio.savemat(name, {'pred': pred})

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


