from architecture import *
from cassi_utils import *
import torch
import torch.nn as nn
import scipy.io as scio
import time
import os
import numpy as np
import logging
from option import opt, diffusion_opt

# init mask
# mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10, opt.in_channels)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
# model
model = model_generator(opt, opt.method, opt.pretrained_model_path, diffusion_opt)

my_summary(model)

