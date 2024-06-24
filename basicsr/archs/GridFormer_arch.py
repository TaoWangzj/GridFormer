"""
file: gridformer_arch.py
about: model for GridFormer
author: Tao Wang
date: 06/29/22


"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY


import numbers
import torch.utils.checkpoint as cp
from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



## Compact Self-Attention (CSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, sample_rate):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim//2, dim//2 * 3, kernel_size=1, bias=bias)

        self.sampler = nn.AvgPool2d(1, stride=sample_rate)
        self.kernel_size = sample_rate
        self.patch_size = sample_rate

        self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size=self.kernel_size, padding=(self.kernel_size // sample_rate - 1),
                                            stride=sample_rate, groups=dim, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim//2 * 3, dim//2 * 3, kernel_size=3, stride=1, padding=1, groups=dim//2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sampler(x)

        x1, x2 = x.chunk(2, dim=1)

        b, c, h, w = x1.shape

        ########### produce q1,k1 and v1 from x1 token feature

        qkv_1 = self.qkv_dwconv(self.qkv(x1))
        q1, k1, v1 = qkv_1.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        ########### produce q2,k2 and v2 from x2 token feature

        qkv_2 = self.qkv_dwconv(self.qkv(x2))
        q2, k2, v2 = qkv_2.chunk(3, dim=1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        ####### cross-token self-attention

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v1)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = torch.cat([out1, out2], dim=1)

        out = self.LocalProp(out)

        out = self.project_out(out)

        out = out[:, :, :H, :W]
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,sample_rate=2,with_cp=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,sample_rate)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

        if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


##########################################################################
## Resizing modules
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DownSample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(UpSample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
class TBM(nn.Module):
    def __init__(self, dim, num_blocks, heads, sample_rate=2,ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(TBM, self).__init__()

        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             LayerNorm_type=LayerNorm_type,sample_rate=sample_rate) for i in range(num_blocks)])

    def forward(self, x):
        x = self.transformer(x)

        return x


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, num_blocks=4, heads=1, ffn_expansion_factor=2.66,LayerNorm_type='WithBias', bias=False, sample_rate=2,kernel_size=1):
    super(make_dense, self).__init__()
    self.Transformer = TBM(dim=nChannels, num_blocks=num_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor,bias=bias,LayerNorm_type=LayerNorm_type,sample_rate=sample_rate)
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = self.Transformer(x)
    # out = F.relu(self.conv(x))
    out = self.conv(out)
    out = F.relu(out)
    out = torch.cat((x, out), 1)
    return out

# Residual Dense Transformer Block (RDTB) architecture
class RDTB(nn.Module):
  def __init__(self, nChannels, nDenselayer, num_blocks, heads, sample_rate, growthRate):
    super(RDTB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate,num_blocks,heads, sample_rate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = F.relu(out)
    out = out + x
    return out


class extractor(nn.Module):
    def __init__(self,in_c, n_feat, num_blocks,
                          sample_rate,heads=1):
        super(extractor, self).__init__()
        self.conv_in = OverlapPatchEmbed(in_c, n_feat)
        self.tbm_in = TBM(dim=n_feat, num_blocks=num_blocks,
                          heads=heads,sample_rate=sample_rate)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.tbm_in(x)
        return x

class reconstruct(nn.Module):
    def __init__(self,in_c, n_feat, num_blocks,
                          sample_rate,heads=1,kernel_size=3):
        super(reconstruct, self).__init__()

        self.tbm_out = TBM(dim=in_c, num_blocks=num_blocks,
                          heads=heads,sample_rate=sample_rate)
        self.conv_out = nn.Conv2d(in_c, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
    def forward(self, x):

        x = self.tbm_out(x)
        x = self.conv_out(x)
        return x


@ARCH_REGISTRY.register()
# --- Main model  --- #
class GridFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,dim=48, kernel_size=3, stride=2, height=3, width=6, num_blocks=[2,3,4],growthRate=16,nDenselayer=3,
                 heads=[2, 4, 8], attention=True, windows=4, sample_rate_trunk =[4,2,2], scale=1):
        super(GridFormer, self).__init__()
        self.TBM_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.extractor_module = nn.ModuleDict()
        self.reconstruct_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.dim = dim
        self.patch_size = windows
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, dim * stride ** (height - 1)))),
                                        requires_grad=attention)

        self.conv_in = OverlapPatchEmbed(in_channels, dim)
        self.conv_out = nn.Conv2d(dim, out_channels, kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2)

        # self.TBM_in = TBM(dim=dim, num_blocks=num_blocks[0],
        #                   heads=1,sample_rate=sample_rate_trunk[0])
        # self.TBM_out = TBM(dim, num_blocks[0], heads=1,sample_rate=sample_rate_trunk[0])

        self.TBM_in = RDTB(nChannels=dim, nDenselayer=nDenselayer, num_blocks=num_blocks[0], growthRate=growthRate, heads=1,sample_rate=sample_rate_trunk[0])
        self.TBM_out = RDTB(nChannels=dim, nDenselayer=nDenselayer, num_blocks=num_blocks[0], growthRate=growthRate, heads=1,sample_rate=sample_rate_trunk[0])

        TBM_in_channels = dim
        for i in range(height):
            for j in range(width - 1):
                # self.TBM_module.update({'{}_{}'.format(i, j): TBM(TBM_in_channels, num_blocks[i], heads[i],sample_rate=sample_rate_trunk[i])})
                self.TBM_module.update({'{}_{}'.format(i, j): RDTB(nChannels=TBM_in_channels, nDenselayer=nDenselayer, num_blocks=num_blocks[i], growthRate=growthRate, heads=heads[i],sample_rate=sample_rate_trunk[i])})
            TBM_in_channels *= stride

        _in_channels = dim
        for i in range(
                height - 1):
            for j in range(width // 2):
                # self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
                if j==0:
                    self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
                    self.extractor_module.update({'{}_{}'.format(i, j): extractor(in_c=in_channels, n_feat=_in_channels*2,num_blocks=num_blocks[i],sample_rate=sample_rate_trunk[i])})
                else:
                    self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1,
                       -1):
            for j in range(width // 2, width):

                if j ==width-1:
                    self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
                    self.reconstruct_module.update({'{}_{}'.format(i, j): reconstruct(in_c=_in_channels, n_feat=out_channels,num_blocks=num_blocks[i],sample_rate=sample_rate_trunk[i])})    ########### tail 3x3 convolution
                else:
                    self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})

                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride


    def forward_features(self, x):

        Image_index = [ 0 for _ in range(self.height)]
        Image_index[0] = x
        for i in range(1,self.height):
            Image_index[i] = F.interpolate(Image_index[i-1], scale_factor=0.5, mode='bilinear',recompute_scale_factor=True,align_corners=False)


        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        # x_index[0][0] = self.conv_in(x)
        x_index[0][0] = self.TBM_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.TBM_module['{}_{}'.format(0, j - 1)](x_index[0][j - 1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i - 1, 0)](x_index[i - 1][0]) + self.extractor_module['{}_{}'.format(i-1, 0)](Image_index[i])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2 ** (i - 1) * self.stride * self.dim)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.TBM_module[
                    '{}_{}'.format(i, j - 1)](x_index[i][j - 1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module[
                                    '{}_{}'.format(i - 1, j)](x_index[i - 1][j])

        x_index[i][j + 1] = self.TBM_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.TBM_module['{}_{}'.format(i, j - 1)](x_index[i][j - 1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i - 1) * self.stride * self.dim)
            x_index[i][k + 1] = self.coefficient[i, k + 1, 0, :channel_num][None, :, None, None] * self.TBM_module[
                '{}_{}'.format(i, k)](x_index[i][k]) + \
                                self.coefficient[i, k + 1, 1, :channel_num][None, :, None, None] * self.upsample_module[
                                    '{}_{}'.format(i, k + 1)](x_index[i + 1][k + 1])

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.dim)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.TBM_module[
                    '{}_{}'.format(i, j - 1)](x_index[i][j - 1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module[
                                    '{}_{}'.format(i, j)](x_index[i + 1][j])

        image_out = [0 for _ in range(self.height)]

        image_out[0] = self.conv_out(self.TBM_out(x_index[i][j])) + Image_index[0]

        for i in range(1,self.height):
            image_out[i] = self.reconstruct_module['{}_{}'.format(i-1,j)](x_index[i][j])+Image_index[i]



        return image_out

    def forward(self, x):
        x = self.forward_features(x)

        x.reverse()
        return x



