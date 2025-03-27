import math
import warnings
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from einops import rearrange

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

class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1.
        output[input < 0] = 0.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply

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
            h = self.conv1(x)
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

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out

class Plane_Guide_Mechanism(nn.Module):  # Spatial_Attention_Block
    def __init__(self, times=8):
        super(Plane_Guide_Mechanism, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(2, 2 * times, 3, 1, 1),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(2 * times, 2 * times, 3, 1, 1),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(2 * times, 1, 3, 1, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x_max = torch.max(x, 1)[0].unsqueeze(1)
        x_mean = torch.mean(x, 1).unsqueeze(1)
        x_compress = torch.cat([x_max, x_mean], dim=1)
        x_out = self.mlp(x_compress)
        return x_out

class Point_Guide_Mechanism(nn.Module):
    def __init__(self, channel):
        super(Point_Guide_Mechanism, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(channel, 1, 1, padding=0, bias=True),
            nn.ReLU())

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_avg_mlp = self.mlp(x_avg) + 1e-6
        x_max_mlp = self.mlp(x_max) + 1e-6
        alpha = x_avg_mlp + x_max_mlp
        return alpha

# def shift_back_4d(inputs, step=2): # input [bs,28,256,310]  output [bs, 28, 256, 256]
#     [bs, nC, row, col] = inputs.shape
#     output = torch.zeros(bs, nC, row, row).cuda().float()
#     for i in range(nC):
#         output[:, i, :, :] = inputs[:, i, :, step * i:step * i + col - (nC - 1) * step]
#     return output

class HyperParamNet(nn.Module):
    def __init__(self, channel=64, times=4, estimate_alpha=True, estimate_beta=True):
        super(HyperParamNet, self).__init__()
        self.estimate_alpha = estimate_alpha
        self.estimate_beta = estimate_beta
        self.fusion = nn.Sequential(nn.Conv2d(56, 28 * times, 3, 1, 1, bias=True, groups=28),
                                    nn.ReLU(),
                                    nn.Conv2d(28 * times, channel, 1, 1, 0, bias=True)
                                    )

        if self.estimate_alpha == True and self.estimate_beta == True:
            self.alpha_fusion = nn.Sequential(Point_Guide_Mechanism(channel=channel // 2))
            self.beta_fusion = nn.Sequential(Plane_Guide_Mechanism())
        elif self.estimate_alpha == True and self.estimate_beta == False:
            self.alpha_fusion = nn.Sequential(Point_Guide_Mechanism(channel=channel))
        elif self.estimate_alpha == False and self.estimate_beta == True:
            self.beta_fusion = nn.Sequential(Plane_Guide_Mechanism())
        else:
            self.alpha_fusion = None
            self.beta_fusion = None


    def forward(self, y, mask_3d_batch):
        bs, ch, h, w = y.shape
        temp = torch.zeros([bs, ch * 2, h, h]).cuda()
        i = 0
        j = 0
        for index in range(ch * 2):
            if index % 2 == 0:
                temp[:, index:index + 1, :, :] = y[:, i:i + 1, :, :]
                i = i + 1
            else:
                temp[:, index:index + 1, :, :] = mask_3d_batch[:, j:j + 1, :, :]
                j = j + 1
        temp = self.fusion(temp)
        _, c, h, w = temp.shape

        if self.estimate_alpha==True and self.estimate_beta==True:
            # estimate alpha
            alpha = self.alpha_fusion(temp[:, :c//2, :, :])
            # estimate beta
            beta = self.beta_fusion(temp[:, c//2:, :, :])
            return alpha, beta
        elif self.estimate_alpha==True and self.estimate_beta==False:
            # estimate alpha
            alpha = self.alpha_fusion(temp)
            return alpha
        elif self.estimate_alpha==False and self.estimate_beta==True:
            # estimate beta
            beta = self.beta_fusion(temp)
            return beta
        else:
            return None

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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class Spectral_Multihead_Self_Attention(nn.Module):
    def __init__(self, dim=224, dim_head=28, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1), requires_grad=True)
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb1 = nn.Parameter(torch.Tensor(1, heads, dim_head, dim_head))
        trunc_normal_(self.pos_emb1)

        self.dim = dim
        self.relu = nn.ReLU()

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h*w, c)#c=dim
        q_inp = self.to_q(x)#(b,hw,dim_head*heads)
        k_inp = self.to_k(x)#(b,hw,dim_head*heads)
        v_inp = self.to_v(x)#(b,hw,dim_head*heads)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        #b=batch_size,h=num_heads,n=hw,d=dim_head

        # q: b,heads,hw,d
        q = q.transpose(-2, -1)#[b,heads,d,hw]
        k = k.transpose(-2, -1)#[b,heads,d,hw]
        v = v.transpose(-2, -1)#[b,heads,d,hw]
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q, A=[b,heads,d,d]
        attn = attn * self.rescale
        attn = attn + self.pos_emb1
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose [b, hw, heads, d]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out = self.proj(x).view(b, h, w, c)

        return out

class Hybrid_Spatial_Multihead_Self_Attention(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=8, type_branch='non_local', patch_size=256):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.type_branch = type_branch

        # position embedding
        if type_branch == 'local':
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        elif type_branch == 'hybrid':
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            h, w = patch_size//self.heads, patch_size//self.heads
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)
        else:
            seq_l1 = window_size[0] * window_size[1]
            h, w = 256 // self.heads, 256 // self.heads
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
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.type_branch == 'local':
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
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
        elif self.type_branch == 'hybrid':
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 = q1 * self.scale
            sim1 = torch.einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = torch.einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = torch.einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = torch.einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            q2 = q
            k2 = k
            v2 = v

            # non-local branch
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
        return out

class Spectral_Self_Attention_Block(nn.Module):
    def __init__(self, dim=224, dim_head=28, heads=8, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Spectral_Multihead_Self_Attention(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)#[b,h,w,c]
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Hybrid_Spatial_Self_Attention_Block(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=64, heads=8, num_blocks=1, type_branch='non_local', patch_size=256):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Hybrid_Spatial_Multihead_Self_Attention(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, type_branch=type_branch, patch_size=patch_size)),
                PreNorm(dim, FeedForward(dim=dim))
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

class Transformer(nn.Module):
    def __init__(self, in_channels=28, unet_stages=3, spatial_sab_type='non_local', patch_size=256):
        super().__init__()
        self.in_channels=in_channels
        self.unet_stages=unet_stages
        # Input projection
        self.embedding = nn.Conv2d(self.in_channels+1, self.in_channels, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_per_stage = self.in_channels
        for i in range(unet_stages):
            if i == 0:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    DoubleResblock(dim_per_stage),
                    nn.Conv2d(dim_per_stage, dim_per_stage * 2, 4, 2, 1, bias=False),  # for feature downsample
                ]))
            else:
                self.encoder_layers.append(nn.ModuleList([
                    # The channels of each head is equal to self.in_channels
                    Spectral_Self_Attention_Block(dim=dim_per_stage, num_blocks=1, dim_head=self.in_channels,
                                                  heads=dim_per_stage // self.in_channels),
                    nn.Conv2d(dim_per_stage, dim_per_stage * 2, 4, 2, 1, bias=False)
                ]))
            dim_per_stage *= 2

        # Bottleneck
        self.bottleneck = Hybrid_Spatial_Self_Attention_Block(dim=dim_per_stage, dim_head=in_channels, heads=dim_per_stage // in_channels, num_blocks=1, type_branch=spatial_sab_type, window_size=(16, 16), patch_size=patch_size)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(unet_stages):
            if i == unet_stages-1:
                self.decoder_layers.append(nn.ModuleList([
                    nn.ConvTranspose2d(dim_per_stage, dim_per_stage // 2, stride=2, kernel_size=2, padding=0,
                                       output_padding=0),  # Upsample
                    nn.Conv2d(dim_per_stage // 2 * 3, dim_per_stage // 2, 1, 1, bias=False),
                    # fusion with encoder output
                    nn.Conv2d(dim_per_stage, dim_per_stage // 2, 1, 1, bias=False),
                    DoubleResblock(dim_per_stage // 2),
                ]))
            else:
                self.decoder_layers.append(nn.ModuleList([
                    nn.ConvTranspose2d(dim_per_stage, dim_per_stage // 2, stride=2, kernel_size=2, padding=0,
                                       output_padding=0),  # Upsample
                    nn.Conv2d(dim_per_stage // 2 * 3, dim_per_stage // 2, 1, 1, bias=False),
                    # fusion with encoder output
                    nn.Conv2d(dim_per_stage, dim_per_stage // 2, 1, 1, bias=False),
                    Hybrid_Spatial_Self_Attention_Block(dim=dim_per_stage//2, dim_head=in_channels,
                                                        heads=(dim_per_stage // 2) // in_channels, num_blocks=1,
                                                        type_branch=spatial_sab_type, window_size=(16, 16), patch_size=patch_size)
                ]))

            dim_per_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, bias=False)

        # activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, noise_level_map, pre_decoder):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(torch.cat([x, noise_level_map], dim=1))

        # Encoder
        fea_encoder = []
        # masks = []
        for (SAB, FeaDownSample) in self.encoder_layers:
            fea = SAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        fea_decoder = []
        # Decoder
        for i, (FeaUpSample, Fusion_tri, Fusion, SAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea_enc = fea_encoder[self.unet_stages - 1 - i]

            diffY = fea_enc.size()[2] - fea.size()[2]
            diffX = fea_enc.size()[3] - fea.size()[3]
            if diffY != 0 or diffX != 0:
                print('Padding for size mismatch ( Enc:', fea_enc.size(), 'Dec:', fea.size(), ')')
                fea = F.pad(fea, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')

            fea_decoder.append(fea)
            if len(pre_decoder) != 0:
                fea = Fusion_tri(torch.cat([fea, fea_enc, pre_decoder[i]], dim=1))
            else:
                fea = Fusion(torch.cat([fea, fea_enc], dim=1))
            # fea = Fusion(torch.cat([fea, fea_enc], dim=1))
            fea = SAB(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out, fea_decoder

class MaskLearning(nn.Module):
    def __init__(self, in_channels=28, features=64):
        super(MaskLearning, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(),
                                   ResBlock(features, 'no_norm'),
                                   ResBlock(features, 'no_norm'),
                                   ResBlock(features, 'no_norm'),
                                   nn.Conv2d(features, in_channels, 3, 1, 1, bias=False),
                                   nn.LeakyReLU()
                                   )

    def forward(self, x):
        out = self.block(x)
        return out

#based complete7
class My_model(nn.Module):
    def __init__(self, in_channels=28, stages=4, unet_stages=3, patch_size=256, optim_mask=True, step=2):
        super(My_model, self).__init__()
        self.in_channels = in_channels
        self.stages = stages
        self.patch_size = patch_size
        self.optim_mask = optim_mask
        self.step = step

        if self.optim_mask == True:
            ## Mask Initialization ##
            self.Phi_batch = Parameter(
                torch.zeros(1, self.in_channels, self.patch_size, self.patch_size),
                requires_grad=True)
            # torch.nn.init.normal_(self.Phi_batch, mean=0, std=0.1)

            self.delta = Parameter(torch.zeros(1), requires_grad=True)
            # torch.nn.init.normal_(self.delta, mean=0, std=0.01)

            # Mask Learning
            self.mask_learning = MaskLearning(in_channels=self.in_channels, features=64)

        self.Phi = Parameter(torch.ones(self.patch_size, self.patch_size), requires_grad=True)
        torch.nn.init.normal_(self.Phi, mean=0, std=0.1)

        #hyperParam estimate
        self.relu = nn.ReLU()
        self.transformers = nn.ModuleList([])
        for i in range(stages):
            self.transformers.append(Transformer(in_channels=self.in_channels, unet_stages=unet_stages, spatial_sab_type='non_local', patch_size=self.patch_size))
        self.fusion = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.para_estimator = HyperParamNet(channel=64, times=4, estimate_alpha=True, estimate_beta=True)

    def y2x(self, y, ch=28, is_norm=False):
        ##  Spilt operator
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]

        sz = y.size()
        if is_norm:
            y = y / ch * 2
        x = torch.zeros([bs, ch, sz[2], sz[2]]).cuda()
        for t in range(ch):
            temp = y[:, :, :, 0 + 2 * t : sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def x2y(self, x):
        ##  Shift and Sum operator
        sz = x.size()
        if len(sz) == 3:
            x = x.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[2]+2*27]).cuda()
        for t in range(28):
            y[:, :, :, 2 * t : sz[2] + 2 * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 2 * t : sz[2] + 2 * t]
        return y

    def A(self, x, Phi):
        temp = x * Phi #[bs,ch,h,w]=[bs,ch,h,w]*[bs,ch,h,w]
        y = self.x2y(temp)#[bs,1,h,w+(ch-1)*step]
        # temp = self.shift(x) * Phi #[bs,ch,h,w+(ch-1)*step]
        # y = torch.sum(temp, dim=1, keepdim=True) #[bs,1,h,w+(ch-1)*step]
        return y

    def At(self, y, Phi):
        x = y * Phi #[bs,ch,h,w]
        return x

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

    def forward(self, input_image, input_mask=None):
        if self.optim_mask == True:
            input_image = input_image.unsqueeze(1)
            bs, _, h, w = input_image.shape
            assert self.patch_size == h and self.patch_size == h
            ch = 28
            meas = input_image

            mask = MyBinarize(self.Phi)  # (h,w)
            mask_3d_batch_ori = mask.contiguous().view(1, 1, h, h)  # (1,1,h,w)
            mask_3d_batch_ori = mask_3d_batch_ori.repeat(bs, ch, 1, 1).cuda()  # (bs, ch, h, w)
            # Phi_batch_ori = self.shift_4d(mask_3d_batch_ori)

            mask_3d_batch_res = self.Phi_batch.repeat(bs, 1, 1, 1).cuda()
            # Phi_batch_res = self.shift_4d(mask_3d_batch_res)

            # Phi_batch = Phi_batch_ori + Phi_batch_res
            mask_3d_batch = mask_3d_batch_ori + mask_3d_batch_res


            # # mask_3d_batch = self.shift_back_4d(Phi_batch)
            # Phi_s_batch = torch.sum(Phi_batch ** 2, dim=1, keepdim=True)  # (batch_size, 1, H, W+(ch-1)*2)
            # Phi_s_batch[Phi_s_batch == 0] = 1

            # y = meas / ch * 2
            # bs, _, row, col = y.shape
            z = self.y2x(meas, is_norm=True)
            z = self.fusion(torch.cat([z, mask_3d_batch_ori], dim=1))
            # z = self.shift_4d(z)

            # flag_Phi = torch.ones(bs, ch, h, h, requires_grad=False).cuda()
            # flag_Phi_batch = self.shift_4d(flag_Phi)
            # meas_ori = None
            # meas_res = None
        else:
            bs, ch, h, w = input_mask.shape
            assert self.patch_size == h and self.patch_size == w
            assert self.in_channels == ch
            meas = input_image.unsqueeze(1)
            mask = MyBinarize(self.Phi)  # (h,w)
            mask_3d_batch = mask.contiguous().view(1, 1, h, h)  # (1,1,h,w)
            mask_3d_batch = mask_3d_batch.repeat(bs, ch, 1, 1).cuda()  # (bs, ch, h, w)
            z = self.y2x(meas, ch, is_norm=True)
            z = self.fusion(torch.cat([z, mask_3d_batch], dim=1))
            Phi_batch = self.shift_4d(mask_3d_batch)
            Phi_s_batch = torch.sum(Phi_batch ** 2, dim=1, keepdim=True)  # (batch_size, 1, H, W+(ch-1)*2)
            Phi_s_batch[Phi_s_batch == 0] = 1

        feature_list = []

        for i in range(self.stages):
            if self.optim_mask == True:
                # generate pre-stage measurement
                # z = self.shift_4d(z)
                # temp_shift = Phi_batch * z
                # meas_pre = torch.sum(temp_shift, dim=1, keepdim=True).cuda()  # (bs,1,h,h+(ch-1)*2)

                temp = mask_3d_batch * z
                meas_pre = self.x2y(temp)

                meas_minu = meas_pre - meas
                meas_minu = self.y2x(meas_minu, is_norm=False)

                # update Phi_batch, refine shift stage
                Phi_batch_g = mask_3d_batch_res - self.delta * meas_minu * z
                mask_3d_batch_res = self.mask_learning(Phi_batch_g)
                mask_3d_batch = mask_3d_batch_ori + mask_3d_batch_res
                Phi_batch = self.shift_4d(mask_3d_batch)
                Phi_s_batch = torch.sum(Phi_batch ** 2, dim=1, keepdim=True)  # (batch_size, 1, H, W+(ch-1)*2)
                Phi_s_batch[Phi_s_batch == 0] = 1

            alpha, beta = self.para_estimator(z, mask_3d_batch)
            Phi_z = self.A(z, mask_3d_batch)
            x = z + self.At(self.y2x(torch.div(meas - Phi_z, alpha + Phi_s_batch), is_norm=False), mask_3d_batch)
            # x = self.shift_back_4d(x)
            z, feature_list = self.transformers[i](x, beta, feature_list)
            # z = self.shift_4d(z)

        temp_shift = mask_3d_batch * z
        meas_pre = self.x2y(temp_shift)
        # meas_pre = torch.sum(temp_shift, dim=1, keepdim=True).cuda()  # (bs,1,h,h+(ch-1)*2)
        # z = self.shift_back_4d(z)

        return z, (mask_3d_batch_ori, mask_3d_batch_res, meas_pre)
