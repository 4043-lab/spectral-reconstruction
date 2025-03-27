import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


from torch.nn import GELU
from einops import rearrange
class Spectral_AttnBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=8,
            norm_groups=16
    ):
        super().__init__()
        num_heads = dim // dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.norm = nn.GroupNorm(norm_groups, dim)
        self.to_q = nn.Linear(dim, dim_head * num_heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * num_heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * num_heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Linear(dim_head * num_heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        b, c, h, w = x_in.shape
        norm = self.norm(x_in)
        x = norm.permute(0, 2, 3, 1).reshape(b, h*w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        # q: b,heads,hw,d
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # q, k, v: b, heads, d, hw
        attn_spe = (k @ q.transpose(-2, -1))   # (d×HW) * (HW×d)
        attn_spe = attn_spe * self.rescale
        attn_spe = attn_spe.softmax(dim=-1)
        attended_values = attn_spe @ v   # b,heads,d,hw
        attended_values = attended_values.permute(0, 3, 1, 2)    # Transpose
        attended_values = attended_values.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(attended_values).view(b, h, w, c).permute(0, 3, 1, 2)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        out = out_c + out_p + x_in

        return out


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


class Local_Spatial_Multihead_Self_Attention(nn.Module):
    def __init__(self, dim, window_size=(8, 8), heads=8, norm_groups=32):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size

        self.norm = nn.GroupNorm(norm_groups, dim)
        # position embedding
        seq_l = window_size[0] * window_size[1]
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)  # dim_head ** -0.5
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape
        norm = self.norm(x)
        norm = norm.permute(0, 2, 3, 1)
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        x_inp = rearrange(norm, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        q = self.to_q(x_inp)
        k = self.to_k(x_inp)
        v = self.to_v(x_inp)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        out = out.permute(0, 3, 1, 2) + x

        return out


class Nonlocal_Spatial_Multihead_Self_Attention(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=8, patch_size=256):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size

        seq_l1 = window_size[0] * window_size[1]
        h, w = patch_size // self.heads, patch_size // self.heads
        seq_l2 = h * w // seq_l1
        self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l2, seq_l2))
        trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)  # dim_head ** -0.5
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q2 = q
        k2 = k
        v2 = v

        # non-local
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                             b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q2, k2, v2))
        q2 = q2 * self.scale
        sim2 = torch.einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = torch.einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = out2.permute(0, 2, 1, 3)
        out = self.to_out(out2)
        out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        out = out.permute(0, 3, 1, 2)

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
            # self.attn = Local_Spatial_Multihead_Self_Attention(dim=dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class TrainableResnetBlocWithAttnEnc(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.trainable_res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            # self.trainable_attn = Local_Spatial_Multihead_Self_Attention(dim=dim_out, norm_groups=norm_groups)
            # self.trainable_attn = Spectral_AttnBlock(dim=dim_out, norm_groups=norm_groups)
            self.trainable_attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.trainable_res_block(x, time_emb)
        if(self.with_attn):
            x = self.trainable_attn(x)
        return x

class TrainableResnetBlocWithAttnMid(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.trainable_res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            # self.trainable_attn = Local_Spatial_Multihead_Self_Attention(dim=dim_out, norm_groups=norm_groups)
            # self.trainable_attn = Spectral_AttnBlock(dim=dim_out, norm_groups=norm_groups)
            self.trainable_attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.trainable_res_block(x, time_emb)
        if(self.with_attn):
            x = self.trainable_attn(x)
        return x

class TrainableResnetBlocWithAttnDec(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.trainable_res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            # self.trainable_attn = Local_Spatial_Multihead_Self_Attention(dim=dim_out, norm_groups=norm_groups)
            # self.trainable_attn = Spectral_AttnBlock(dim=dim_out, norm_groups=norm_groups)
            self.trainable_attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.trainable_res_block(x, time_emb)
        if(self.with_attn):
            x = self.trainable_attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=2,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            # print("use attn:{}".format(use_attn))
            channel_mult = inner_channel * channel_mults[ind]
            for i in range(0, res_blocks):
                if i == res_blocks-1:
                    downs.append(TrainableResnetBlocWithAttnEnc(
                        pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=False))
                else:
                    downs.append(ResnetBlocWithAttn(
                        pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout,
                        with_attn=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult

            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False),
            TrainableResnetBlocWithAttnMid(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                                dropout=dropout, with_attn=True)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for i in range(0, res_blocks+1):
                if i == res_blocks:
                    ups.append(TrainableResnetBlocWithAttnDec(
                        pre_channel + feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout,
                        norm_groups=norm_groups, with_attn=False))
                else:
                    ups.append(ResnetBlocWithAttn(
                        pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=False))

                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn) or isinstance(layer, TrainableResnetBlocWithAttnEnc):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn) or isinstance(layer, TrainableResnetBlocWithAttnMid):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn) or isinstance(layer, TrainableResnetBlocWithAttnDec):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
