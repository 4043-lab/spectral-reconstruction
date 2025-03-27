import torch
from .My_model_PixelShuffle import My_model

def model_generator(opt, method, pretrained_model_path=None, isTrain=True):
    if method == 'my_model':
        model = My_model(in_channels=opt.in_channels, unet_stages=3, num_blocks=[1, 1, 1], patch_size=256, step=2)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)

    return model