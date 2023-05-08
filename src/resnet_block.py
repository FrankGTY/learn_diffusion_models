from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from functools import partial
from helpers import *
import torch
import torch.nn.functional as F
from torch import nn


class WeightStandarizedConv2d(nn.Conv2d):
    '''
    Weight standarized purpotedly works synergistticaly with group normalization
    '''
    
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 ", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 ", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()
        
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandarizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
            
        x = self.act(x)
        return x
    

class ResnetBlock(nn.Module):
    """
    ResnetBlock 类被定义为 nn.Module 类的子类，它是 PyTorch 中所有神经网络模块的基类。该类接受三个参数：dim、dim_out 和 time_emb_dim。dim 参数指定输入通道的数量，dim_out 指定输出通道的数量，time_emb_dim 指定时间嵌入的维度，这是一个可选的输入。

    ResnetBlock 类包含三个主要组件：MLP、两个块和一个残差卷积。MLP 是一个可选组件，用于为两个块中的批量归一化层生成比例和偏移参数。两个块是 Block 类的实例，该类在代码的其他地方定义。残差卷积用于匹配输入和输出张量的维度，如果它们不同的话。

    ResnetBlock 类的 forward 方法接受两个参数：x 和 time_emb。x 参数是输入张量，time_emb 参数是一个可选的时间嵌入张量。如果 MLP 存在并且提供了时间嵌入，则使用 MLP 为两个块中的批量归一化层生成比例和偏移参数。然后将输入张量通过两个块，将第二个块的输出添加到输入张量的残差卷积的输出中。然后返回结果张量。
    """
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2))
            if exists(time_emb_dim)
            else None
        )
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
            
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)