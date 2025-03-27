import torch.nn as nn
from einops import rearrange
# from mamba_ssm import Mamba
import torch
import torch.nn.functional as F
from .Mamba_cross import Mamba

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class GFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net_up = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        )
        self.net_low = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.SiLU()
        )
        self.toOut = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = x.permute(0, 3, 1, 2)
        out_up = self.net_up(x)
        out_low = self.net_low(x)
        out = self.toOut(out_up * out_low)

        return out.permute(0, 2, 3, 1)


import scipy.io as sio
class MambaAttn(nn.Module):
    def __init__(self, dim, patch_size):
        super(MambaAttn, self).__init__()
        '''
        Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        '''
        self.window_size=5
        self.mamba = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, input_h=patch_size, input_w=patch_size)
        self.mamba_l = Mamba(dim, expand=1, d_state=2, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, input_h=self.window_size, input_w=self.window_size)
        self.toOut = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape

        # global mamba
        x_mamba = x.view(b, h * w, c)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.view(b, h, w, c)

        # local mamba
        x_mamba_l = rearrange(x, 'b (hnum win0) (wnum win1) c -> (b hnum wnum) (win0 win1) c', win0=self.window_size, win1=self.window_size, hnum=h//self.window_size, wnum=w//self.window_size)
        x_mamba_l = self.mamba_l(x_mamba_l)
        x_mamba_l = rearrange(x_mamba_l, '(b hnum wnum) (win0 win1) c -> b (hnum win0) (wnum win1) c', win0=self.window_size, win1=self.window_size, hnum=h//self.window_size, wnum=w//self.window_size)

        out = x_mamba + x_mamba_l
        out = self.toOut(out)
        return out


class MambaBlock(nn.Module):
    def __init__(self, dim, patch_size, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, MambaAttn(dim=dim, patch_size=patch_size)),
                PreNorm(dim, GFFN(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class DualMambaNet(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1], patch_size=256):
        super(DualMambaNet, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales - 1):
            self.encoder_layers.append(nn.ModuleList([
                MambaBlock(dim=dim_scale, num_blocks=num_blocks[i], patch_size=patch_size),
                nn.Conv2d(dim_scale, dim_scale * 2, 2, 2, bias=False)
            ]))
            dim_scale *= 2
            patch_size //= 2

        # Bottleneck
        self.bottleneck = MambaBlock(dim=dim_scale, num_blocks=num_blocks[-1], patch_size=patch_size)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales - 1):
            patch_size *= 2
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                MambaBlock(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], patch_size=patch_size),
            ]))
            dim_scale //= 2
            

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        # hb, wb = 16, 16
        # pad_h = (hb - h_inp % hb) % hb
        # pad_w = (wb - w_inp % wb) % wb
        # x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:,:28,:,:]

        # Encoder
        fea_encoder = []

        for (MB, FeaDownSample) in self.encoder_layers:
            fea = MB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, MB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales - 2 - i]], dim=1))
            fea = MB(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out[:, :, :h_inp, :w_inp]