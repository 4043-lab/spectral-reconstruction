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
from .Mamba4d import Mamba4DNet


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

    def forward(self, x):
        out = self.block1(x) + x
        return out

import scipy.io as sio
class My_model_onlyDiff(nn.Module):
    def __init__(self, num_iterations, diffusion_opt, isFinetune=False, patch_size=256, batch_size=1, in_channel=28):
        super(My_model_onlyDiff, self).__init__()
        self.isFinetune = isFinetune
        # self.fusion = nn.Conv2d(29, 28, 1, padding=0, bias=True)
        # self.resblocks = nn.Sequential(
        #     Resblock(28),
        #     Resblock(28),
        #     Resblock(28)
        # )
        self.num_iterations = num_iterations
        # self.rows = nn.ParameterList([])
        # for i in range(num_iterations):
        #     if i == num_iterations - 1:
        #         self.rows.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
        #     else:
        #         self.rows.append(nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))

        self.diffusion_prior = networks.define_G(diffusion_opt)

        self.n_timestep = diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['n_timestep']
        self.register_buffer('betas', torch.linspace(diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['linear_start'], diffusion_opt['model']['beta_schedule'][diffusion_opt['phase']]['linear_end'], self.n_timestep).double())
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()

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

    def unfolding(self, i, z, v, Phi, Phi_s, y, mask):
        T = v + self.rows[i]
        T = self.shift_4d(T)
        Phi_T = A(T, Phi)
        x = T + At(torch.div(y - Phi_T, self.rows[i] + Phi_s), Phi)
        x = self.shift_back_4d(x)

        # z, z_dec = self.deep_prior[i](x, z, z_dec)
        # z = self.deep_prior[i](x, Phi)

        return x

    @torch.no_grad()
    def unfolding_no_grad(self, i, z, v, Phi, Phi_s, y):
        T = v + self.rows[i]
        T = self.shift_4d(T)
        Phi_T = A(T, Phi)
        x = T + At(torch.div(y - Phi_T, self.rows[i] + Phi_s), Phi)
        x = self.shift_back_4d(x)

        # z, z_dec = self.deep_prior[i](x, z, z_dec)
        # z = self.deep_prior[i](x)

        return x

    def forward(self, y, input_image, input_mask, Phi, Phi_s, cond):
        """
        :param y: [b,256,310]
        :param input_image: [b, 28, 256, 256]
        :param input_mask: [b, 28, 256, 256]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        device = input_image.device
        bs, ch, h, w = input_image.shape
        mask = input_mask[:, 0:1, :, :]
        # z_dec = None
        um = mask

        # z = self.fusion(torch.cat([input_image, mask], dim=1))
        # z = self.resblocks(z)
        # z = input_image
        v = torch.randn(input_image.shape, device=device)
        # v = z
        c = self.n_timestep // self.num_iterations
        ddim_timestep_seq = np.asarray(list(range(0, self.n_timestep, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # zs = []
        conditions = torch.cat([cond, mask], dim=1)

        for i in reversed(range(self.num_iterations)):
            if self.isFinetune:
                # z = self.unfolding(i, z, v, Phi, Phi_s, y, um)
                v, rm = self.p_sample_predict_mask(v, conditions, ddim_timestep_seq[i], ddim_timestep_prev_seq[i])
            else:
                # z = self.unfolding(i, z, v, Phi, Phi_s, y, um)
                v, rm = self.p_sample_no_grad_predict_mask(v, conditions, ddim_timestep_seq[i], ddim_timestep_prev_seq[i])

            um = mask  + rm

            v_numpy = v.detach().cpu().numpy()
            rm_numpy = rm.detach().cpu().numpy()
            um_numpy = um.detach().cpu().numpy()
            sio.savemat("./visualization/v_{}.mat".format(i), {'v':v_numpy})
            sio.savemat("./visualization/rm_{}.mat".format(i), {'rm':rm_numpy})
            sio.savemat("./visualization/um_{}.mat".format(i), {'um':um_numpy})

            um_4d = um.repeat(1, ch, 1, 1).cuda()
            Phi = self.shift_4d(um_4d)
            Phi_s = torch.sum(Phi ** 2, dim=1, keepdim=False)  # (batch_size, H, W+(ch-1)*2)
            Phi_s[Phi_s == 0] = 1

        return v, um


