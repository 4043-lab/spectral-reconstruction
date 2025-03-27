import torch.nn as nn
from einops import rearrange
# from mamba_ssm import Mamba
from .Mamba_cross import Mamba
import torch
import torch.nn.functional as F


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


class EFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.toIn = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.net_up = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.net_low = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.toOut = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = x.permute(0, 3, 1, 2)
        x = self.toIn(x)
        out_up = self.net_up(x)
        out_low = self.net_low(x)
        out = self.toOut(out_up * out_low)

        return out.permute(0, 2, 3, 1)


class MaskGuidedMechanism(nn.Module):
    def __init__(self, n_feat):
        super(MaskGuidedMechanism, self).__init__()
        # self.conv1 = nn.Conv2d(n_feat, n_feat*4, kernel_size=1, bias=True)
        # self.depth_conv = nn.Conv2d(n_feat*4, n_feat*4, kernel_size=5, padding=2, bias=True, groups=n_feat)
        # self.conv2 = nn.Conv2d(n_feat*4, 1, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def shift_back(self, inputs, step=2):  # input [bs,28,256,310]  output [bs, 28, 256, 256]
        [bs, nC, row, col] = inputs.shape
        down_sample = 256 // row
        step = float(step) / float(down_sample * down_sample)
        out_col = row
        for i in range(nC):
            inputs[:, i, :, :out_col] = inputs[:, i, :, int(step * i):int(step * i) + out_col]
        return inputs[:, :, :, :out_col]

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = self.shift_back(mask_shift)
        return mask_emb


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
        self.mamba = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, input_h=patch_size, input_w=patch_size)
        self.act = nn.SiLU()
        self.toMask = nn.Linear(dim, dim, bias=False)
        self.toOut = nn.Linear(dim, dim, bias=False)
        self.mm = MaskGuidedMechanism(dim)

    def forward(self, x, mask):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape

        x_mask = self.act(self.toMask(x))

        x_mamba = x.view(b, h * w, c)
        # x_mamba = rearrange(x_mamba, 'b hh ww c -> b (hh ww) c', hh=h, ww=w)
        x_mamba = self.mamba(x_mamba)
        # x_mamba = rearrange(x_mamba, 'b (hh ww) c -> b hh ww c', hh=h, ww=w)
        x_mamba = x_mamba.view(b, h, w, c)

        maskAttn = self.mm(mask).permute(0, 2, 3, 1)

        out = x_mamba + x_mask * maskAttn
        out = self.toOut(out)
        return out

class Mamba4DNet(nn.Module):
    def __init__(self, dim, patch_size, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, MambaAttn(dim=dim, patch_size=patch_size)),
                PreNorm(dim, GFFN(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, mask) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out