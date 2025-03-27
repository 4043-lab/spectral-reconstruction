import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class UpSample(nn.Module):
    # Given input channel and scale, upsample the input x to shape_size
    def __init__(self, scale, channel):
        super(UpSample, self).__init__()
        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=scale, stride=scale))

    def forward(self, x, shape_size):
        yh, yw = shape_size
        out = self.transpose(x)
        xh, xw = out.size()[-2:]
        h = yh - xh
        w = yw - xw
        pt = h // 2
        pb = h - pt
        pl = w // 2
        pr = w - w // 2
        out = F.pad(out, (pl, pr, pt, pb), mode='reflect')
        return out

class UpSample_Pixel_Shuffle(nn.Module):
    # Given input channel and scale, upsample the input x to shape_size
    def __init__(self, scale, channel):
        super(UpSample_Pixel_Shuffle, self).__init__()
        self.channel = channel // (scale**2)
        self.upsample = nn.PixelShuffle(scale)
        self.conv = nn.Conv2d(self.channel, channel, 1, 1, 0, bias=False)
        self.transpose = nn.Sequential(
            # nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            # nn.GELU(),
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False),
            nn.GELU(),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        )
        # self.relu = nn.ReLU()

    def forward(self, x, shape_size):
        yh, yw = shape_size
        x = self.conv(self.upsample(x))
        out = self.transpose(x) + x
        return out


class DownSample(nn.Module):
    def __init__(self, scale, channel):
        super(DownSample, self).__init__()
        self.scale = scale
        self.downsample = nn.AvgPool2d(scale)
        self.conv = nn.Conv2d(channel, channel * scale,3,1,1)

    def forward(self, x):
        out = self.downsample(x)
        out = self.conv(out)
        return out


class DownSample_Pixel_Unshuffle(nn.Module):
    def __init__(self, scale, channel):
        super(DownSample_Pixel_Unshuffle, self).__init__()
        self.channel = channel * (scale**2)
        self.downsample = nn.PixelUnshuffle(scale)
        self.conv = nn.Conv2d(self.channel, self.channel // 4, 1, 1, 0, bias=False)
        self.transform = nn.Sequential(
            # nn.Conv2d(self.channel // 4, self.channel // 4, 1, 1, 0, bias=False),
            # nn.GELU(),
            nn.Conv2d(self.channel // 4, self.channel // 4, 3, 1, 1, bias=False, groups=self.channel // 4),
            nn.GELU(),
            nn.Conv2d(self.channel // 4, self.channel // 4, 1, 1, 0, bias=False)
        )
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.downsample(x)
        out = self.conv(out)
        out = self.transform(out) + out
        return out


class ResBlock(nn.Module):
    def __init__(self, num_channel, acti='ReLU'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.conv2 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU()
        self.acti = acti

    def forward(self, x):
        if self.acti == 'ReLU':
            h = self.relu(self.bn1(self.conv1(x)))
            h = self.relu(self.bn2(self.conv2(h)))
            h = h + x
        elif self.acti == 'no_norm':
            h = self.relu(self.conv1(x))
            h = self.conv2(h)
            h = h + x
        else:
            h = self.relu(self.conv1(x))
            h = self.conv2(h)
            h = h + x
        return h


class DoubleResblock(nn.Module):
    def __init__(self, num_channel):
        super(DoubleResblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x, temp=None):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=7, memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor),
        )
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        # b, n, c = x.shape
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        x = x.view(b, h*w, c)
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return out


class Conv_Spatial_Attention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.conv_q = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwConv7 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=7, stride=1, padding=3, bias=False,
                                 groups=in_ch)
        self.proj = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gelu = nn.GELU()

    def forward(self, inputs):
        q = self.gelu(self.conv_q(inputs))
        v = self.conv_v(inputs)
        dw_q = self.dwConv7(q)
        out = self.proj(dw_q * v)
        return out


class Single_Reference_Spatial_Attention(nn.Module):
    def __init__(self, inplanes):
        super(Single_Reference_Spatial_Attention, self).__init__()
        self.inplanes = inplanes
        self.inter_planes = inplanes

        self.conv_q = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)   #g
        # self.conv_k = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_v = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)   #theta
        # self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q, mode='fan_in')
        kaiming_init(self.conv_v, mode='fan_in')

        self.conv_q.inited = True
        self.conv_v.inited = True

    # HR spatial attention
    def spatial_attention(self, x):
        q_spa = self.conv_q(x)
        # v_spa = self.conv_v(x)
        batch, channel, height, width = q_spa.size()

        k_spa = self.avg_pool(x)
        batch, channel, k_spa_h, k_spa_w = k_spa.size()

        k_spa = k_spa.view(batch, channel, k_spa_h * k_spa_w).permute(0, 2, 1)
        q_spa = q_spa.view(batch, channel, height * width)

        attn_spa = torch.matmul(k_spa, q_spa)
        # attn_spa = self.softmax(attn_spa)
        attn_spa = attn_spa.view(batch, 1, height, width)
        attn_spa = self.conv_out(attn_spa)
        attn_spa = self.sigmoid(attn_spa)

        out = attn_spa * x

        return out

    def forward(self, x):
        out = self.spatial_attention(x)

        return out


class Single_Reference_Spectral_Attention(nn.Module):
    def __init__(self, inplanes):
        super(Single_Reference_Spectral_Attention, self).__init__()
        self.inplanes = inplanes
        self.inter_planes = inplanes

        self.conv_q_spe = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_k_spe = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_v_spe = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(self.inter_planes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        # self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True

    # HR spectral attention
    def spectral_attention(self, x):
        q_spe = self.conv_q_spe(x)
        k_spe = self.conv_k_spe(x)
        # v_spe = self.conv_v_spe(x)

        batch, channel, height, width = q_spe.size()

        q_spe = q_spe.view(batch, channel, height * width)
        q_spe = F.normalize(q_spe, dim=-1, p=2)

        k_spe = k_spe.view(batch, 1, height * width)
        k_spe = F.normalize(k_spe, dim=-1, p=2)

        attn_spe = torch.matmul(q_spe, k_spe.transpose(1,2))
        attn_spe = attn_spe.unsqueeze(-1)
        attn_spe = self.conv_out(attn_spe)
        attn_spe = self.sigmoid(attn_spe)

        out = x * attn_spe

        return out

    def forward(self, x):
        out = self.spectral_attention(x)

        return out


class Mix_Single_Reference_Spectral_Spatial_Attention_Layer(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.inter_planes = inplanes

        self.conv_q = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_k_spe = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v_spe = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v_spa = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(self.inter_planes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_spe = nn.Softmax(dim=2)
        self.softmax_spa = nn.Softmax(dim=1)
        self.conv_norm = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.maskguide = MaskGuidedMechanism(self.inplanes, self.inter_planes)

    def forward(self, x, mask):
        q = self.conv_q(x)
        v_spe = self.conv_v_spe(x)
        v_spa = self.conv_v_spa(x)
        mg = self.maskguide(mask) * v_spa

        batch, channel, height, width = q.size()
        q = q.view(batch, channel, height * width)

        # Spectral Attention
        k_spe = self.conv_k_spe(x)
        k_spe = k_spe.view(batch, 1, height * width)
        k_spe = self.softmax_spe(k_spe)

        attn_spe = torch.matmul(q, k_spe.transpose(1, 2))
        attn_spe = attn_spe.unsqueeze(-1)
        attn_spe = self.conv_up(attn_spe)
        attn_spe_norm = self.softmax_spa(attn_spe)

        out_spe = v_spe * attn_spe_norm

        #Spatial Attention
        k_spa = attn_spe_norm
        batch, channel, k_spa_h, k_spa_w = k_spa.size()
        k_spa = k_spa.view(batch, channel, k_spa_h * k_spa_w).permute(0, 2, 1)

        attn_spa = torch.matmul(k_spa, q)
        # attn_spa_norm = self.softmax_spe(attn_spa)
        attn_spa_norm = attn_spa.view(batch, 1, height, width)
        attn_spa_norm = self.sigmoid(self.conv_norm(attn_spa_norm))

        out_spa = attn_spa_norm * v_spa

        out = self.conv_out(out_spa + out_spe + mg)

        return out


class Dense_MixS_Attention_Layer(nn.Module):
    def __init__(self, inchannels, split_num=4, kernel_size=1):
        super().__init__()
        self.split_num = split_num
        self.block_inch = inchannels // split_num
        self.block_last_inch = inchannels - (split_num - 1) * self.block_inch
        self.inchannels = inchannels

        self.attn_layer = nn.ModuleList([])
        self.block_inchs = []
        for i in range(split_num - 1):
            self.attn_layer.append(Mix_Single_Reference_Spectral_Spatial_Attention_Layer(self.block_inch))
            self.block_inchs.append(self.block_inch)
        self.attn_layer.append(Mix_Single_Reference_Spectral_Spatial_Attention_Layer(self.block_last_inch))
        self.block_inchs.append(self.block_last_inch)

        self.conv_out = nn.Conv2d(self.inchannels, self.inchannels, kernel_size, 1, kernel_size//2, bias=False)

    def forward(self, x, mask):
        Ins = x.split(self.block_inchs, 1)
        Masks = mask.split(self.block_inchs, 1)
        outs = []
        for i in range(self.split_num):
            x_out = self.attn_layer[i](Ins[i], Masks[i])
            outs.append(x_out)

        out = torch.cat(outs, dim=1)
        out = self.conv_out(out) + out
        return out


class Co_Spatial_Multihead_Self_Attention(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=8, ustg=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size

        # position embedding
        seq_l = window_size[0] * window_size[1]
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        ag_inchannels = dim_head*(2**ustg)
        self.ag = nn.Sequential(nn.Conv2d(ag_inchannels, dim, 1, 1, 0, bias=False), nn.AdaptiveAvgPool2d(window_size))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)  # dim_head ** -0.5
        self.relu = nn.ReLU()

    def forward(self, x, y):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        gk = self.ag(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        gk_inp = rearrange(gk, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        q = self.to_q(x_inp)
        k = self.to_k(gk_inp)
        v = self.to_v(x_inp)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, v = map(lambda t: rearrange(t, '(c b) h n d -> c b h n d', c=(h//w_size[0])**2), (q, v))
        q = q * self.scale
        sim = torch.einsum('c b h i d, b h j d -> c b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = torch.einsum('c b h i j, c b h j d -> c b h i d', attn, v)
        out = rearrange(out, 'c b h n d -> (c b) n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])

        return out


# self-calibration
class SelfCalibration(nn.Module):
    def __init__(self, channel):
        super(SelfCalibration, self).__init__()
        self.conin = nn.Conv2d(channel, channel, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.sq = nn.Sequential(
            nn.Linear(16, 4, False),
            nn.LeakyReLU(True),
            nn.Linear(4, 16),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1),
        )
        self.act = nn.Sigmoid()
        self.conout = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conin(x)
        avg = self.pooling(x).view(b * c, -1)
        cat = self.sq(avg).view(b, c, -1)
        sat = self.sa(x).view(b, -1, h * w)
        att = self.act(torch.bmm(cat, sat).view(x.size()))
        att = att * x
        out = self.conout(att)
        return out


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, pooling_r=2):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=padding, bias=False),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=padding, bias=False),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class CoAttention(nn.Module):
    def __init__(self, scale, channel, ratio):
        super(CoAttention, self).__init__()
        self.scale = scale
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.sq = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        out = self.pooling(x).view(b, c)
        out = self.sq(out).view(b, c, 1, 1)
        out = out * x
        out = torch.sum(out.view(b, self.scale, c // self.scale, h, w), dim=1, keepdim=False)
        return out


class CoAttention_MS(nn.Module):
    def __init__(self, scale, channel):
        super().__init__()
        self.inplanes = channel * scale
        self.inter_planes = channel
        self.scale = scale

        ustg = scale
        heads = self.inter_planes // 28
        ag_inchannels = 28 * (2 ** ustg)
        dim = self.inter_planes
        self.ag_up = UpSample(2 ** (ustg - heads + 1), ag_inchannels)
        self.ag_attn = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
                                     nn.ReLU(),
                                     nn.Conv2d(dim, 1, 1, 1, 0, bias=False))

        # self.conv_q_spe = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_k_spe = nn.Conv2d(self.inter_planes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_v_spe = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_spe = nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax_spe = nn.Softmax(dim=2)
        self.softmax_spa = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ag):
        q_spe = x
        batch, channel, height, width = q_spe.size()
        q_spe = q_spe.view(batch, channel, height * width)

        ag_up = self.ag_up(ag, [height, width])
        ag_map = self.ag_attn(ag_up)

        k_spe = ag_map.view(batch, 1, height * width)
        k_spe = self.softmax_spe(k_spe)

        attn_spe = torch.matmul(q_spe, k_spe.transpose(1, 2))
        attn_spe = attn_spe.unsqueeze(-1)
        attn_spe = self.conv_up(attn_spe)
        attn_spe_norm = self.softmax_spa(attn_spe)

        out_spe = x * attn_spe_norm * self.sigmoid(ag_map)
        out_spe = torch.sum(out_spe.view(batch, self.scale, channel // self.scale, height, width), dim=1, keepdim=False)
        k_spa = torch.sum(attn_spe.view(batch, self.scale, channel // self.scale, 1, 1), dim=1, keepdim=False)

        out_spe = self.conv_out_spe(out_spe)

        return out_spe, k_spa


class SingleSPA(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.inter_planes = inchannels
        self.conv_q_spa = nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        self.conv_v_spa = nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        self.conv_out_spa = nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.softmax_spe = nn.Softmax(dim=2)
        self.softmax_spa = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, k_spa):
        # spatial
        k_spa = self.softmax_spa(k_spa)
        batch, channel, k_spa_h, k_spa_w = k_spa.size()
        k_spa = k_spa.view(batch, channel, k_spa_h * k_spa_w).permute(0, 2, 1)

        q = self.conv_q_spa(x)
        batch, channel, height, width = q.size()
        q = q.view(batch, channel, height * width)

        v = self.conv_v_spa(x)

        attn_spa = torch.matmul(k_spa, q)
        attn_spa_norm = self.softmax_spe(attn_spa)
        attn_spa_norm = attn_spa_norm.view(batch, 1, height, width)

        out = self.conv_out_spa(attn_spa_norm * v)

        return out


class AdaIN(nn.Module):
    # input: number of channels in the noise estimation map, guidance information
    # channel: input x
    def __init__(self, scale, guide_channel, channel):
        super(AdaIN, self).__init__()
        self.scale = scale
        if scale != 0:
            self.up = UpSample(scale, guide_channel)
            self.con1 = nn.Conv2d(guide_channel // scale, channel, 1, 1, 0)
        else:
            self.con1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False)
        self.con1_f = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.con2 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False)
        self.con3 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False)

    def forward(self, de_map, guide_info):
        _, c, h, w = de_map.size()
        if self.scale != 0:
            guide_info = self.up(guide_info, de_map.size()[-2:])
        # normlize
        mu = torch.mean(de_map.view(-1, c, h * w), dim=2)[:, :, None, None]
        sigma = torch.std(de_map.view(-1, c, h * w), dim=2)[:, :, None, None] + 10e-5
        de_map = (de_map - mu) / sigma

        guide_info = self.con1(guide_info)
        guide_info = self.con1_f(guide_info)
        gama = self.con2(guide_info)
        beta = self.con3(guide_info)

        de_map = de_map * gama + beta

        return de_map


class AdaINBlock(nn.Module):
    def __init__(self, scale, guide_channel, channel):
        super(AdaINBlock, self).__init__()
        self.adain1 = AdaIN(scale, guide_channel, channel)
        self.con1 = nn.Conv2d(channel, channel, 1, 1, 0)

    def forward(self, demap, guide_info):
        # x = self.con(demap)
        x = self.adain1(demap, guide_info)
        x = self.con1(x)

        return demap + x


class CASC(nn.Module):
    def __init__(self, scale, inchannels, ratio=4):
        super(CASC, self).__init__()
        self.scale = scale
        self.CA = CoAttention(scale, inchannels * scale, ratio)
        self.SC = SelfCalibration(inchannels)

    def forward(self, x):
        fm = self.CA(x)
        out = self.SC(fm)

        return out


class CASC_MS(nn.Module):
    def __init__(self, scale, inchannels, ratio=4):
        super(CASC_MS, self).__init__()
        self.scale = scale
        self.CA = CoAttention_MS(scale, inchannels * scale)
        # self.spa = Single_Reference_Spatial_Attention_Block(inchannels)

    def forward(self, x, ag):
        out_spe, attn = self.CA(x, ag)
        # out = self.spa(out_spe, attn)

        return out_spe


class MSCASC(nn.Module):
    def __init__(self, n_scale, middle, ratio=4):
        super(MSCASC, self).__init__()
        self.n_scale = n_scale
        self.sample_dict = nn.ModuleDict()

        for i in range(n_scale):
            for j in range(n_scale):
                if i < j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': DownSample(2 ** (j - i), middle[i])})
                if i > j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': UpSample(2 ** (i - j), middle[i])})

    def select_sample(self, x, shape_size, i, j):
        if i == j:
            return x
        else:
            if i > j:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x, shape_size)
            else:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x)

    def forward(self, x):
        res = []
        for i in range(self.n_scale):
            # 尺度归一
            fuse = [self.select_sample(x[j], x[i].size()[-2:], j, i) for j in range(self.n_scale)]
            res.append(torch.cat(fuse, dim=1))

        return res


class ConcatLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatLayer, self).__init__()
        self.clayer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        out = self.clayer(x)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        out = self.net(x)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.fn(x, *args, **kwargs)


class FeatureFusion(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.sq = nn.Sequential(
            nn.Linear(channels * 2, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels * 2),
            nn.Sigmoid()
        )
        # self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        # self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        b, c, h, w = x.shape
        attn = self.pooling(x).view(b, c)
        attn = self.sq(attn).view(b, c, 1, 1)
        out = attn * x
        out = torch.sum(out.view(b, 2, c // 2, h, w), dim=1, keepdim=False)
        # attn_1, attn_2 = attn.chunk(2, dim=1)
        # out1 = self.softmax(self.conv1(attn_1)) * x1
        # out2 = self.softmax(self.conv2(attn_2)) * x2
        # out = out1 + out2
        return out


class Mix_Single_Reference_Spatial_Spectral_Attention_Block(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Mix_Single_Reference_Spectral_Spatial_Attention_Layer(dim)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x
        return out


class Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(nn.Module):
    def __init__(self, dim, heads=4, num_blocks=1, kernel_size=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Dense_MixS_Attention_Layer(dim, split_num=heads, kernel_size=kernel_size)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (attn, ff) in self.blocks:
            x = attn(x, mask) + x
            x = ff(x) + x
        out = x
        return out


class Transformer_head(nn.Module):
    def __init__(self, in_channels=28, unet_stages=2, num_blocks=[1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.unet_stages = unet_stages

        # Input projection
        # self.embedding = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_per_stage = self.in_channels
        for i in range(unet_stages):
            if i == unet_stages-1:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[i],
                                                                                kernel_size=3),
                    # nn.Conv2d(dim_per_stage, dim_per_stage, 4, 2, 1, bias=False)
                    DownSample_Pixel_Unshuffle(2, dim_per_stage)
                ]))
            else:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[i],
                                                                                kernel_size=1),
                    # nn.Conv2d(dim_per_stage, dim_per_stage, 4, 2, 1, bias=False)
                    DownSample_Pixel_Unshuffle(2, dim_per_stage)
                ]))
            # dim_per_stage *= 2

        # Bottleneck
        self.bottleneck = Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4, num_blocks=num_blocks[-1], kernel_size=3)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(unet_stages):
            if i == 0:
                self.decoder_layers.append(nn.ModuleList([
                    # UpSample(2, dim_per_stage),
                    # fusion with encoder output
                    # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
                    UpSample_Pixel_Shuffle(2, dim_per_stage),
                    # AdaINBlock(0, dim_per_stage, dim_per_stage),
                    # FeatureFusion(dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[
                                                                                    self.unet_stages - 1 - i],
                                                                                kernel_size=3)
                ]))
            else:
                self.decoder_layers.append(nn.ModuleList([
                    # UpSample(2, dim_per_stage),
                    # fusion with encoder output
                    # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
                    UpSample_Pixel_Shuffle(2, dim_per_stage),
                    # AdaINBlock(0, dim_per_stage, dim_per_stage),
                    # FeatureFusion(dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[
                                                                                    self.unet_stages - 1 - i],
                                                                                kernel_size=1)
                ]))
            # reverse = not reverse
            # dim_per_stage //= 2

        # Output projection
        # self.mapping = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, bias=False)

    def forward(self, x, mask_guide):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = x# * mask_guide[0]  # self.embedding(x)

        # Encoder
        fea_encoder = []
        fea_decoder = []
        for i, (SAB, FeaDownSample) in enumerate(self.encoder_layers):
            fea = SAB(fea, mask_guide[i])
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)# * mask_guide[i + 1]

        # Bottleneck
        fea = self.bottleneck(fea, mask_guide[-1])
        fea_decoder.append(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, SAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea, fea_encoder[self.unet_stages - 1 - i].size()[-2:])
            fea_enc = fea_encoder[self.unet_stages - 1 - i]
            # fea = AGAI(fea, fea_enc)
            fea = Fusion(torch.cat([fea_enc, fea], dim=1))
            # fea = fea_enc + fea
            # fea = fea * mask_guide[self.unet_stages - i - 1]
            fea = SAB(fea, mask_guide[self.unet_stages - i - 1])
            fea_decoder.append(fea)

        # Mapping
        out = fea + x

        return out, fea_decoder


class Transformer(nn.Module):
    def __init__(self, in_channels=28, unet_stages=2, num_blocks=[1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.unet_stages = unet_stages

        # Input projection
        # self.embedding = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        # self.Ags = nn.ModuleList([])
        dim_per_stage = self.in_channels
        for i in range(unet_stages):
            if i == unet_stages-1:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[i],
                                                                                kernel_size=3),
                    # nn.Conv2d(dim_per_stage, dim_per_stage, 4, 2, 1, bias=False)
                    DownSample_Pixel_Unshuffle(2, dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False)
                    # AdaINBlock(0, dim_per_stage, dim_per_stage)
                ]))
            else:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[i],
                                                                                kernel_size=1),
                    # nn.Conv2d(dim_per_stage, dim_per_stage, 4, 2, 1, bias=False)
                    DownSample_Pixel_Unshuffle(2, dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False)
                    # AdaINBlock(0, dim_per_stage, dim_per_stage)
                ]))
            # self.Ags.append(AdaINBlock(0, dim_per_stage, dim_per_stage))
            # dim_per_stage *= 2
            # self.Ags.append(FeatureFusion(dim_per_stage))

        # Bottleneck
        self.bottleneck = Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4, num_blocks=num_blocks[-1], kernel_size=3)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(unet_stages):
            if i == 0:
                self.decoder_layers.append(nn.ModuleList([
                    # UpSample(2, dim_per_stage),
                    # fusion with encoder output
                    # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
                    UpSample_Pixel_Shuffle(2, dim_per_stage),
                    # AdaINBlock(0, dim_per_stage, dim_per_stage),
                    # FeatureFusion(dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                                num_blocks=num_blocks[
                                                                                    self.unet_stages - 1 - i],
                                                                                kernel_size=3)
                ]))
            else:
                self.decoder_layers.append(nn.ModuleList([
                    # UpSample(2, dim_per_stage),
                    # fusion with encoder output
                    # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
                    UpSample_Pixel_Shuffle(2, dim_per_stage),
                    # AdaINBlock(0, dim_per_stage, dim_per_stage),
                    # FeatureFusion(dim_per_stage),
                    nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
                    Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                        num_blocks=num_blocks[self.unet_stages - 1 - i], kernel_size=1)
            ]))
            # reverse = not reverse
            # dim_per_stage //= 2

        # Output projection
        # self.mapping = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, bias=False)

    def forward(self, x, fea_pre, mask_guide):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = x# * mask_guide[0]

        # Encoder
        fea_encoder = []
        fea_decoder = []

        for i, (SAB, FeaDownSample, Fusion) in enumerate(self.encoder_layers):
            fea = SAB(fea, mask_guide[i])
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            fea = Fusion(torch.cat([fea, fea_pre[self.unet_stages - i - 1]], dim=1))
            # fea = fea + fea_pre[self.unet_stages-i-1]
            # fea = fea * mask_guide[i + 1]

        # Bottleneck
        fea = self.bottleneck(fea, mask_guide[-1])
        fea_decoder.append(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, SAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea, fea_encoder[self.unet_stages - 1 - i].size()[-2:])
            fea_enc = fea_encoder[self.unet_stages - 1 - i]
            # fea = AGAI(fea, fea_enc)
            fea = Fusion(torch.cat([fea_enc, fea], dim=1))
            # fea = fea_enc + fea
            # fea = fea * mask_guide[self.unet_stages - i - 1]
            fea = SAB(fea, mask_guide[self.unet_stages - i - 1])
            fea_decoder.append(fea)

        # Mapping
        out = fea + x

        return out, fea_decoder


class MaskGuidedMechanism(nn.Module):
    def __init__(self, inchannels, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        # mask_emb = shift_back(mask_shift)
        return mask_shift


# based complete7
class My_model(nn.Module):
    def __init__(self, in_channels=28, unet_stages=3, patch_size=256, step=2, num_blocks=[1, 1, 1]):
        super(My_model, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.step = step
        self.unet_stages = unet_stages
        self.nfeats = 48

        self.fusion = nn.Conv2d(self.in_channels * 2, self.nfeats, 1, 1, 0, bias=False)

        self.transformer_1 = Transformer_head(in_channels=self.nfeats, unet_stages=2, num_blocks=[1, 1, 1])
        self.transformer_2 = Transformer(in_channels=self.nfeats, unet_stages=2, num_blocks=[1, 1, 1])
        self.transformer_3 = Transformer(in_channels=self.nfeats, unet_stages=2, num_blocks=[1, 1, 1])

        self.mapping = nn.Conv2d(self.nfeats, self.in_channels, 3, 1, 1, bias=False)

        self.conv = nn.Conv2d(self.in_channels, self.nfeats, 1, 1, 0)
        # self.maskGuide_1 = MaskGuidedMechanism(self.nfeats, self.nfeats)
        self.downSample_1 = nn.Conv2d(self.nfeats, self.nfeats, 2, 2, 0, bias=False)
        # self.maskGuide_2 = MaskGuidedMechanism(self.nfeats, self.nfeats)
        self.downSample_2 = nn.Conv2d(self.nfeats, self.nfeats, 2, 2, 0, bias=False)#DownSample_Pixel_Unshuffle(2, self.nfeats)
        # self.maskGuide_3 = MaskGuidedMechanism(self.nfeats, self.nfeats)

    def forward(self, input_image, input_mask=None):
        if input_mask == None:
            input_mask = torch.zeros((1, 28, 256, 256)).cuda()

        z0 = self.fusion(torch.cat([input_image, input_mask], dim=1))

        input_mask_1 = self.conv(input_mask)
        # mask_guide_1 = self.maskGuide_1(input_mask_1)
        input_mask_2 = self.downSample_1(input_mask_1)
        # mask_guide_2 = self.maskGuide_2(input_mask_2)
        input_mask_3 = self.downSample_2(input_mask_2)
        # mask_guide_3 = self.maskGuide_3(input_mask_3)

        mask = [input_mask_1, input_mask_2, input_mask_3]
        # mask_guide = [mask_guide_1, mask_guide_2, mask_guide_3]

        z1, z1_dec = self.transformer_1(z0, mask)
        z2, z2_dec = self.transformer_2(z1, z1_dec, mask)
        z3, z3_dec = self.transformer_3(z2, z2_dec, mask)

        out = z3 + z0
        out = self.mapping(out)

        return out
