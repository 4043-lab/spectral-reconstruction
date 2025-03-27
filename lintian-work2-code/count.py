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
# from architecture.My_model import *

# init mask
# mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10, opt.in_channels)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# model
model = model_generator(opt, opt.method, opt.pretrained_model_path)

my_summary(model)
