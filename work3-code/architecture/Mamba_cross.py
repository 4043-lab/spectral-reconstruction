# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    """
    bimamba_type:
        none: original mamba
        v0: mamba with two inputs (x, z | B, C, t)
        v1: bidirectional mamba with single input (shared weights)
        v2: bidirectional mamba with single input
        v3: bidirectional mamba with two inputs (x, z | B, C, t)
        v4: bidirectional mamba with two inputs (x | B, C, t, z)
        v5: bidirectional mamba with two inputs (shared weights | x, z | B, C, t)
        v6: cross-scan mamba with single input
        v7: cross-scan mamba with two inputs (x, z | B, C, t)
        v8: cross-scan mamba with single input (shared weights)
        v9: cross-scan mamba with two inputs (shared weights | x, z | B, C, t)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
        use_norm=False,
        input_h=64,
        input_w=64
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.use_norm = use_norm
        self.input_h = input_h
        self.input_w = input_w

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        if bimamba_type in ["v0", "v3", "v5", "v7", "v9"]:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.in_proj_b = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        elif bimamba_type == "v4":
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
            self.in_proj_b = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

        if bimamba_type in ["v2", "v3", "v4", "v6", "v7"]:
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        if bimamba_type in ["v6", "v7", "v8", "v9"]:
            A_c = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_c_log = torch.log(A_c)  # Keep A_c_log in fp32
            self.A_c_log = nn.Parameter(A_c_log)
            self.A_c_log._no_weight_decay = True
            A_d = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_d_log = torch.log(A_d)  # Keep A_d_log in fp32
            self.A_d_log = nn.Parameter(A_d_log)
            self.A_d_log._no_weight_decay = True

        if bimamba_type in ["v6", "v7"]:
            self.conv1d_c = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.conv1d_d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_c = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.x_proj_d = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_d = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_d = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_c._no_weight_decay = True
            self.D_d._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        if use_norm:
            self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, hidden_states, extra_emb=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        extra_emb: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        if extra_emb is not None:
            extra_emb = rearrange(
                self.in_proj_b.weight @ rearrange(extra_emb, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj_b.bias is not None:
                extra_emb = extra_emb + rearrange(self.in_proj_b.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v0":
                x, z = xz.chunk(2, dim=1)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                x_dbl = self.x_proj(rearrange(extra_emb, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )
                y = rearrange(y, "b d l -> b l d")
                if self.use_norm:
                    y = self.norm(y)
                out = self.out_proj(y)

            elif self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(out + out_b.flip([-1]), "b d l -> b l d")), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), 
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, 
                                       self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type in ["v3", "v4"]:
                # xz for main body, extra_emb for B, C, t
                # forward ssm
                if self.bimamba_type == "v3":
                    x, z = xz.chunk(2, dim=1)
                    x_b, z_b = xz.flip([-1]).chunk(2, dim=1)
                    extra = extra_emb
                    extra_b = extra_emb.flip([-1])
                else:
                    x = xz
                    extra, z = extra_emb.chunk(2, dim=1)
                    x_b = xz.flip([-1])
                    extra_b, z_b = extra_emb.flip([-1]).chunk(2, dim=1)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                x_dbl = self.x_proj(rearrange(extra, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # backward ssm
                A_b = -torch.exp(self.A_b_log.float())
                if causal_conv1d_fn is None:
                    x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x_b = causal_conv1d_fn(
                        x=x_b,
                        weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                        bias=self.conv1d_b.bias,
                        activation=self.activation,
                    )
                x_dbl_b = self.x_proj_b(rearrange(extra_b, "b d l -> (b l) d"))  # (bl d)
                dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt_b = self.dt_proj_b.weight @ dt_b.t()
                dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
                B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y_b = selective_scan_fn(
                    x_b,
                    dt_b,
                    A_b,
                    B_b,
                    C_b,
                    self.D_b.float(),
                    z=z_b,
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # combination
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]), "b d l -> b l d")), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]), "b d l -> b l d"), 
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]), "b d l -> b l d") / 2), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]), "b d l -> b l d") / 2, 
                                       self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v5":
                # xz for main body, extra_emb for B, C, t
                A_b = -torch.exp(self.A_b_log.float())
                # forward ssm
                x, z = xz.chunk(2, dim=1)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                x_dbl = self.x_proj(rearrange(extra_emb, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # backward ssm
                y_b = selective_scan_fn(
                    x.flip([-1]),
                    dt.flip([-1]),
                    A_b,
                    B.flip([-1]),
                    C.flip([-1]),
                    self.D.float(),
                    z=z.flip([-1]),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # combination
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]), "b d l -> b l d")), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]), "b d l -> b l d"), 
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]), "b d l -> b l d") / 2), 
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]), "b d l -> b l d") / 2, 
                                       self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v6":
                xz_b = rearrange(xz, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                xz_b = rearrange(xz_b.transpose(-2, -1).contiguous(),
                                 "b d w h -> b d (w h)",
                                 w=self.input_w,
                                 h=self.input_h)
                A_b = -torch.exp(self.A_b_log.float())
                A_c = -torch.exp(self.A_c_log.float())
                A_d = -torch.exp(self.A_d_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_c = mamba_inner_fn_no_out_proj(
                    xz_b,
                    self.conv1d_c.weight,
                    self.conv1d_c.bias,
                    self.x_proj_c.weight,
                    self.dt_proj_c.weight,
                    A_c,
                    None,
                    None,
                    self.D_c.float(),
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                )
                out_d = mamba_inner_fn_no_out_proj(
                    xz_b.flip([-1]),
                    self.conv1d_d.weight,
                    self.conv1d_d.bias,
                    self.x_proj_d.weight,
                    self.dt_proj_d.weight,
                    A_d,
                    None,
                    None,
                    self.D_d.float(),
                    delta_bias=self.dt_proj_d.bias.float(),
                    delta_softplus=True,
                )
                out_c = rearrange(out_c, "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                out_c = rearrange(out_c.transpose(-2, -1).contiguous(),
                                  "b d h w -> b d (h w)",
                                  h=self.input_h,
                                  w=self.input_w)
                out_d = rearrange(out_d.flip([-1]), "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                out_d = rearrange(out_d.transpose(-2, -1).contiguous(),
                                  "b d h w -> b d (h w)",
                                  h=self.input_h,
                                  w=self.input_w)
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(out + out_b.flip([-1]) + out_c + out_d, "b d l -> b l d")),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(out + out_b.flip([-1]) + out_c + out_d, "b d l -> b l d"),
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(out + out_b.flip([-1]) + out_c + out_d, "b d l -> b l d") / 4),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(out + out_b.flip([-1]) + out_c + out_d, "b d l -> b l d") / 4,
                                       self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v7":
                x, z = xz.chunk(2, dim=1)
                x_b, z_b = xz.flip([-1]).chunk(2, dim=1)
                extra = extra_emb
                extra_b = extra_emb.flip([-1])
                xz_b = rearrange(xz, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                xz_b = rearrange(xz_b.transpose(-2, -1).contiguous(),
                                 "b d w h -> b d (w h)",
                                 w=self.input_w,
                                 h=self.input_h)
                x_c, z_c = xz_b.chunk(2, dim=1)
                x_d, z_d = xz_b.flip([-1]).chunk(2, dim=1)
                extra_emb_b = rearrange(extra_emb, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                extra_emb_b = rearrange(extra_emb_b.transpose(-2, -1).contiguous(),
                                        "b d w h -> b d (w h)",
                                        w=self.input_w,
                                        h=self.input_h)
                extra_c = extra_emb_b
                extra_d = extra_emb_b.flip([-1])

                # direction 1
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                x_dbl = self.x_proj(rearrange(extra, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # direction 2
                A_b = -torch.exp(self.A_b_log.float())
                if causal_conv1d_fn is None:
                    x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x_b = causal_conv1d_fn(
                        x=x_b,
                        weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                        bias=self.conv1d_b.bias,
                        activation=self.activation,
                    )
                x_dbl_b = self.x_proj_b(rearrange(extra_b, "b d l -> (b l) d"))  # (bl d)
                dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt_b = self.dt_proj_b.weight @ dt_b.t()
                dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
                B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y_b = selective_scan_fn(
                    x_b,
                    dt_b,
                    A_b,
                    B_b,
                    C_b,
                    self.D_b.float(),
                    z=z_b,
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # direction 3
                A_c = -torch.exp(self.A_c_log.float())
                if causal_conv1d_fn is None:
                    x_c = self.act(self.conv1d_c(x_c)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x_c = causal_conv1d_fn(
                        x=x_c,
                        weight=rearrange(self.conv1d_c.weight, "d 1 w -> d w"),
                        bias=self.conv1d_c.bias,
                        activation=self.activation,
                    )
                x_dbl_c = self.x_proj_c(rearrange(extra_c, "b d l -> (b l) d"))  # (bl d)
                dt_c, B_c, C_c = torch.split(x_dbl_c, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt_c = self.dt_proj_c.weight @ dt_c.t()
                dt_c = rearrange(dt_c, "d (b l) -> b d l", l=seqlen)
                B_c = rearrange(B_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C_c = rearrange(C_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y_c = selective_scan_fn(
                    x_c,
                    dt_c,
                    A_c,
                    B_c,
                    C_c,
                    self.D_c.float(),
                    z=z_c,
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # direction 4
                A_d = -torch.exp(self.A_d_log.float())
                if causal_conv1d_fn is None:
                    x_d = self.act(self.conv1d_d(x_d)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x_d = causal_conv1d_fn(
                        x=x_d,
                        weight=rearrange(self.conv1d_d.weight, "d 1 w -> d w"),
                        bias=self.conv1d_d.bias,
                        activation=self.activation,
                    )
                x_dbl_d = self.x_proj_d(rearrange(extra_d, "b d l -> (b l) d"))  # (bl d)
                dt_d, B_d, C_d = torch.split(x_dbl_d, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt_d = self.dt_proj_d.weight @ dt_d.t()
                dt_d = rearrange(dt_d, "d (b l) -> b d l", l=seqlen)
                B_d = rearrange(B_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C_d = rearrange(C_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y_d = selective_scan_fn(
                    x_d,
                    dt_d,
                    A_d,
                    B_d,
                    C_d,
                    self.D_d.float(),
                    z=z_d,
                    delta_bias=self.dt_proj_d.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # combination
                y_c = rearrange(y_c, "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                y_c = rearrange(y_c.transpose(-2, -1).contiguous(),
                                "b d h w -> b d (h w)",
                                h=self.input_h,
                                w=self.input_w)
                y_d = rearrange(y_d.flip([-1]), "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                y_d = rearrange(y_d.transpose(-2, -1).contiguous(),
                                "b d h w -> b d (h w)",
                                h=self.input_h,
                                w=self.input_w)
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d")),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d"),
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d") / 4),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d") / 4,
                                       self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type in ["v8", "v9"]:
                A_b = -torch.exp(self.A_b_log.float())
                A_c = -torch.exp(self.A_c_log.float())
                A_d = -torch.exp(self.A_d_log.float())
                x, z = xz.chunk(2, dim=1)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )
                if self.bimamba_type == "v8":
                    x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                else:
                    x_dbl = self.x_proj(rearrange(extra_emb, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]

                xz_b = rearrange(xz, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                xz_b = rearrange(xz_b.transpose(-2, -1).contiguous(),
                                 "b d w h -> b d (w h)",
                                 w=self.input_w,
                                 h=self.input_h)
                x_b, z_b = xz_b.chunk(2, dim=1)
                dt_b = rearrange(dt, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                dt_b = rearrange(dt_b.transpose(-2, -1).contiguous(),
                                 "b d w h -> b d (w h)",
                                 w=self.input_w,
                                 h=self.input_h)
                B_b = rearrange(B, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                B_b = rearrange(B_b.transpose(-2, -1).contiguous(),
                                "b d w h -> b d (w h)",
                                w=self.input_w,
                                h=self.input_h)
                C_b = rearrange(C, "b d (h w) -> b d h w", h=self.input_h, w=self.input_w)
                C_b = rearrange(C_b.transpose(-2, -1).contiguous(),
                                "b d w h -> b d (w h)",
                                w=self.input_w,
                                h=self.input_h)

                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )
                y_b = selective_scan_fn(
                    x.flip([-1]),
                    dt.flip([-1]),
                    A_b,
                    B.flip([-1]),
                    C.flip([-1]),
                    self.D.float(),
                    z=z.flip([-1]),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )
                y_c = selective_scan_fn(
                    x_b,
                    dt_b,
                    A_c,
                    B_b,
                    C_b,
                    self.D.float(),
                    z=z_b,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )
                y_d = selective_scan_fn(
                    x_b.flip([-1]),
                    dt_b.flip([-1]),
                    A_d,
                    B_b.flip([-1]),
                    C_b.flip([-1]),
                    self.D.float(),
                    z=z_b.flip([-1]),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

                # combination
                y_c = rearrange(y_c, "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                y_c = rearrange(y_c.transpose(-2, -1).contiguous(),
                                "b d h w -> b d (h w)",
                                h=self.input_h,
                                w=self.input_w)
                y_d = rearrange(y_d.flip([-1]), "b d (w h) -> b d w h", w=self.input_w, h=self.input_h)
                y_d = rearrange(y_d.transpose(-2, -1).contiguous(),
                                "b d h w -> b d (h w)",
                                h=self.input_h,
                                w=self.input_w)
                if not self.if_devide_out:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d")),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d"),
                                       self.out_proj.weight, self.out_proj.bias)
                else:
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d") / 4),
                                       self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y + y_b.flip([-1]) + y_c + y_d, "b d l -> b l d") / 4,
                                       self.out_proj.weight, self.out_proj.bias)

            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        if self.init_layer_scale is not None:
                out = out * self.gamma
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)