import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import model.networks as networks
import numpy as np
from .HST import HST
from .MST import MST
from .LMSFormer import LMSFormer, LMSFormer_head
from .MambaNet import MambaNet


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().cuda()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out

from .Mamba4d import Mamba4DNet

class My_model_onlyMamba(nn.Module):
    def __init__(self, num_iterations, diffusion_opt, isFinetune=False, patch_size=256):
        super(My_model_onlyMamba, self).__init__()
        self.isFinetune = isFinetune
        self.fusion = nn.Conv2d(29, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.deep_prior = nn.ModuleList([])
        # self.fusions = nn.ModuleList([])
        self.mius = nn.ParameterList([])
        # self.rows = nn.ParameterList([])
        # self.row = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        for i in range(num_iterations):
            if i == num_iterations - 1:
                self.deep_prior.append(
                    # MST(),
                    # HST(in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1]),
                    # LMSFormer_head(),
                    MambaNet(in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1], patch_size=patch_size)
                    # Mamba4DNet(dim=28, patch_size=patch_size, num_blocks=5)
                )
                # self.fusions.append(nn.Sequential(nn.Conv2d(56, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                #                 Resblock(64), Resblock(64),
                #                 nn.Conv2d(64, 28, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()))
                self.mius.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
                # self.rows.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
            else:
                self.deep_prior.append(
                    # MST(),
                    # HST(in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1]),
                    # LMSFormer(),
                    MambaNet(in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1], patch_size=patch_size)
                    # Mamba4DNet(dim=28, patch_size=patch_size, num_blocks=5)
                )
                # self.fusions.append(nn.Sequential(nn.Conv2d(56, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                #                                   Resblock(64),
                #                                   nn.Conv2d(64, 28, kernel_size=3, stride=1, padding=1),
                #                                   nn.LeakyReLU()))
                self.mius.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
                # self.rows.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))

        # self.diffusion_prior = networks.define_G(diffusion_opt)
        # self.diffusion_prior.load_state_dict(torch.load("experiments/cassi_dauhst3stg_spa_l2loss_alltrainable_ddpm_240727_120756/checkpoint/I80000_E80_gen.pth"), strict=False)
        # for k, v in self.diffusion_prior.named_parameters():
        #     if 'trainable' in k:
        #         v.requires_grad = True
        #     else:
        #         v.requires_grad = False

        # self.n_timestep = diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['n_timestep']
        # self.register_buffer('betas', torch.linspace(diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['linear_start'], diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['linear_end'], self.n_timestep).double())
        # alphas = 1. - self.betas
        # self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()

    def shift_4d(self, inputs, step=2):  # input [bs,28,256,256]  output [bs, 28, 256, 310]
        [bs, nC, row, col] = inputs.shape
        output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
        return output

    def shift_back_4d(self, inputs, step=2): # input [bs,28,256,310]  output [bs, 28, 256, 256]
        [bs, nC, row, col] = inputs.shape
        output = torch.zeros(bs, nC, row, row).cuda().float()
        for i in range(nC):
            output[:, i, :, :] = inputs[:, i, :, step * i:step * i + col - (nC - 1) * step]
        return output

    # @torch.no_grad()
    def p_sample(self, xt, conditions, ddim_timestep_seq, ddim_timestep_prev_seq):
        device=xt.device
        batch_size, _, _, _ = xt.shape
        t = torch.full((batch_size,), ddim_timestep_seq, device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), ddim_timestep_prev_seq, device=device, dtype=torch.long)
        # 1. get current and previous alpha_cumprod
        alpha_cumprod_t = extract(self.alphas_cumprod, t, xt.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, xt.shape)

        # 2. predict noise using model
        pred_noise = self.diffusion_prior.denoise_fn(torch.cat([conditions, xt], dim=1), t)

        # 3. get the predicted x_0
        pred_x0 = (xt - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

        # 5. compute "direction pointing to x_t" of formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise

        # 6. compute x_{t-1} of formula (12)
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

        return x_prev

    @torch.no_grad()
    def p_sample_no_grad(self, xt, conditions, ddim_timestep_seq, ddim_timestep_prev_seq):
        device = xt.device
        batch_size, _, _, _ = xt.shape
        t = torch.full((batch_size,), ddim_timestep_seq, device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), ddim_timestep_prev_seq, device=device, dtype=torch.long)
        # 1. get current and previous alpha_cumprod
        alpha_cumprod_t = extract(self.alphas_cumprod, t, xt.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, xt.shape)

        # 2. predict noise using model
        pred_noise = self.diffusion_prior.denoise_fn(torch.cat([conditions, xt], dim=1), t)

        # 3. get the predicted x_0
        pred_x0 = (xt - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

        # 5. compute "direction pointing to x_t" of formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise

        # 6. compute x_{t-1} of formula (12)
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

        return x_prev

    @torch.no_grad()
    def p_sample_no_grad_predict_mask(self, xt, conditions, ddim_timestep_seq, ddim_timestep_prev_seq):
        device = xt.device
        batch_size, _, _, _ = xt.shape
        t = torch.full((batch_size,), ddim_timestep_seq, device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), ddim_timestep_prev_seq, device=device, dtype=torch.long)
        # 1. get current and previous alpha_cumprod
        alpha_cumprod_t = extract(self.alphas_cumprod, t, xt.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, xt.shape)

        # 2. predict noise using model
        pred_noise, update_mask = self.diffusion_prior.denoise_fn(torch.cat([conditions, xt], dim=1), t)

        # 3. get the predicted x_0
        pred_x0 = (xt - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        # pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

        # 5. compute "direction pointing to x_t" of formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise

        # 6. compute x_{t-1} of formula (12)
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

        return x_prev, update_mask

    def p_sample_predict_mask(self, xt, conditions, ddim_timestep_seq, ddim_timestep_prev_seq):
        device = xt.device
        batch_size, _, _, _ = xt.shape
        t = torch.full((batch_size,), ddim_timestep_seq, device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), ddim_timestep_prev_seq, device=device, dtype=torch.long)
        # 1. get current and previous alpha_cumprod
        alpha_cumprod_t = extract(self.alphas_cumprod, t, xt.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, xt.shape)

        # 2. predict noise using model
        pred_noise, update_mask = self.diffusion_prior.denoise_fn(torch.cat([conditions, xt], dim=1), t)

        # 3. get the predicted x_0
        pred_x0 = (xt - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        # pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

        # 5. compute "direction pointing to x_t" of formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_noise

        # 6. compute x_{t-1} of formula (12)
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt

        return x_prev, update_mask

    def unfolding(self, i, z, Phi, Phi_s, y, mask):
        T = self.mius[i] * z
        # T = self.fusions[i](torch.cat([z, v], dim=1))
        T = self.shift_4d(T)
        Phi_T = A(T, Phi)
        x = T + At(torch.div(y - Phi_T, self.mius[i] + Phi_s), Phi)
        x = self.shift_back_4d(x)

        # z, z_dec = self.deep_prior[i](x, z, z_dec)
        z = self.deep_prior[i](x, mask)

        return z

    @torch.no_grad()
    def unfolding_no_grad(self, i, z, v, Phi, Phi_s, y):
        T = self.mius[i] * z
        # T = self.fusions[i](torch.cat([z, v], dim=1))
        T = self.shift_4d(T)
        Phi_T = A(T, Phi)
        x = T + At(torch.div(y - Phi_T, self.mius[i] + Phi_s), Phi)
        x = self.shift_back_4d(x)

        # z, z_dec = self.deep_prior[i](x, z, z_dec)
        z = self.deep_prior[i](x)

        return z

    def forward(self, y, input_image=None, input_mask=None, Phi=None, Phi_s=None, cond=None):
        """
        :param y: [b,256,310]
        :param input_image: [b, 28, 256, 256]
        :param input_mask: [b, 28, 256, 256]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        # device = input_image.device
        # if y is None:
        #     y = torch.zeros((1, 256, 310)).cuda()
        # if input_image is None:
        #     input_image = torch.zeros((1, 28, 256, 256)).cuda()
        # if input_mask is None:
        #     input_mask = torch.zeros((1, 28, 256, 256)).cuda()
        # if Phi is None:
        #     Phi = torch.zeros((1, 28, 256, 310)).cuda()
        # if Phi_s is None:
        #     Phi_s = torch.zeros((1, 256, 310)).cuda()
        # if cond is None:
        #     cond = torch.zeros((1, 28, 256, 256)).cuda()
        bs, ch, h, w = input_image.shape
        mask = input_mask[:, 0:1, :, :]
        # z_dec = None
        um = input_mask

        z = self.fusion(torch.cat([input_image, mask], dim=1))

        for i in reversed(range(self.num_iterations)):
            z = self.unfolding(i, z, Phi, Phi_s, y, um)

        return z, um


