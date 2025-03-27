import torch
from .My_model import My_model
from .DAUHST import DAUHST
from .SPECAT import SPECAT
from .My_model_onlyMamba import My_model_onlyMamba
from .My_model_onlyDiff import My_model_onlyDiff
from .BiSRNet import BiSRNet
from .padut import PADUT
from .DHM import DHM

def model_generator(opt, method, pretrained_model_path=None, diffusion_opt=None):
    if method == 'my_model':
        model = My_model(num_iterations=3, diffusion_opt=diffusion_opt, isFinetune=opt.isFinetune, patch_size=opt.patch_size, inchannels=opt.in_channels)
    elif method == 'my_model_onlyMamba':
        model = My_model_onlyMamba(num_iterations=3, diffusion_opt=diffusion_opt, isFinetune=opt.isFinetune, patch_size=opt.patch_size)
    elif method == 'my_model_onlyDiff':
        model = My_model_onlyDiff(num_iterations=5, diffusion_opt=diffusion_opt, isFinetune=opt.isFinetune, patch_size=opt.patch_size)
    elif 'dauhst' in method:
        model = DAUHST(num_iterations=3)
    elif 'dhm' in method:
        model = DHM(num_iterations=3, patch_size=opt.patch_size)
    elif 'specat' in method:
        model = SPECAT(dim=28, stage=1, num_blocks=[2, 1], attention_type='full')
    elif method == 'bisrnet':
        model = BiSRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1])
    elif 'padut' in method:
        num_iterations = int(method.split('_')[-1])
        model = PADUT(in_c=28, n_feat=28, nums_stages=num_iterations-1)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)

    return model