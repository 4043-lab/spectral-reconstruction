import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
import warnings

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
        # self.maskguide = MaskGuidedMechanism(self.inplanes, self.inter_planes)

    def forward(self, x):
        q = self.conv_q(x)
        v_spe = self.conv_v_spe(x)
        v_spa = self.conv_v_spa(x)
        # mg = self.maskguide(mask) * v_spa

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

        out = self.conv_out(out_spa + out_spe)

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

    def forward(self, x):
        Ins = x.split(self.block_inchs, 1)
        # Masks = mask.split(self.block_inchs, 1)
        outs = []
        for i in range(self.split_num):
            x_out = self.attn_layer[i](Ins[i])
            outs.append(x_out)

        out = torch.cat(outs, dim=1)
        out = self.conv_out(out) + out
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

class LMSFormer_head(nn.Module):
    def __init__(self, in_channels=28, n_feats=32, unet_stages=2, num_blocks=[1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.unet_stages = unet_stages
        self.n_feats = n_feats

        # Input projection
        self.embedding = nn.Conv2d(self.in_channels * 2, self.n_feats, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_per_stage = self.n_feats
        for i in range(unet_stages):
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
            self.decoder_layers.append(nn.ModuleList([
                UpSample_Pixel_Shuffle(2, dim_per_stage),
                nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
                Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
                                                                            num_blocks=num_blocks[
                                                                                self.unet_stages - 1 - i],
                                                                            kernel_size=1)
            ]))
            # reverse = not reverse
            # dim_per_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.n_feats, self.in_channels, 3, 1, 1, bias=False)

    def forward(self, x, z, fea_pre=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(torch.cat([x, z], dim=1))

        # Encoder
        fea_encoder = []
        fea_decoder = []
        for i, (SAB, FeaDownSample) in enumerate(self.encoder_layers):
            fea = SAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)# * mask_guide[i + 1]

        # Bottleneck
        fea = self.bottleneck(fea)
        fea_decoder.append(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, SAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea, fea_encoder[self.unet_stages - 1 - i].size()[-2:])
            fea_enc = fea_encoder[self.unet_stages - 1 - i]
            fea = Fusion(torch.cat([fea_enc, fea], dim=1))
            fea = SAB(fea)
            fea_decoder.append(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out, fea_decoder

class LMSFormer(nn.Module):
    def __init__(self, in_channels=28, n_feats=32, unet_stages=2, num_blocks=[1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.unet_stages = unet_stages
        self.n_feats = n_feats

        # Input projection
        self.embedding = nn.Conv2d(self.in_channels * 2, self.n_feats, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        # self.Ags = nn.ModuleList([])
        dim_per_stage = self.n_feats
        for i in range(unet_stages):
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
            # if i == 0:
            #     self.decoder_layers.append(nn.ModuleList([
            #         # UpSample(2, dim_per_stage),
            #         # fusion with encoder output
            #         # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
            #         UpSample_Pixel_Shuffle(2, dim_per_stage),
            #         # AdaINBlock(0, dim_per_stage, dim_per_stage),
            #         # FeatureFusion(dim_per_stage),
            #         nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
            #         Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
            #                                                                     num_blocks=num_blocks[
            #                                                                         self.unet_stages - 1 - i],
            #                                                                     kernel_size=3)
            #     ]))
            # else:
            #     self.decoder_layers.append(nn.ModuleList([
            #         # UpSample(2, dim_per_stage),
            #         # fusion with encoder output
            #         # AdaINBlock(2**(unet_stages-i), self.in_channels*(2**(unet_stages)), dim_per_stage//2),
            #         UpSample_Pixel_Shuffle(2, dim_per_stage),
            #         # AdaINBlock(0, dim_per_stage, dim_per_stage),
            #         # FeatureFusion(dim_per_stage),
            #         nn.Conv2d(dim_per_stage * 2, dim_per_stage, 1, 1, 0, bias=False),
            #         Dense_Mix_Single_Reference_Spatial_Spectral_Attention_Block(dim=dim_per_stage, heads=4,
            #             num_blocks=num_blocks[self.unet_stages - 1 - i], kernel_size=1)
            # ]))
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
        self.mapping = nn.Conv2d(self.n_feats, self.in_channels, 3, 1, 1, bias=False)

    def forward(self, x, z, fea_pre):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        fea = self.embedding(torch.cat([x, z], dim=1))

        # Encoder
        fea_encoder = []
        fea_decoder = []

        for i, (SAB, FeaDownSample, Fusion) in enumerate(self.encoder_layers):
            fea = SAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            fea = Fusion(torch.cat([fea, fea_pre[self.unet_stages - i - 1]], dim=1))

        # Bottleneck
        fea = self.bottleneck(fea)
        fea_decoder.append(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, SAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea, fea_encoder[self.unet_stages - 1 - i].size()[-2:])
            fea_enc = fea_encoder[self.unet_stages - 1 - i]
            fea = Fusion(torch.cat([fea_enc, fea], dim=1))
            fea = SAB(fea)
            fea_decoder.append(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out, fea_decoder