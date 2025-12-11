# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath, to_2tuple
from functools import partial
from domainbed.mixout.mixlinear import MixLinear
import domainbed.models.loralib as lora

class MixoutMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            lora_kwargs=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        
        if lora_kwargs is not None:
            linear_layer = lora.Linear
        else:
            linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else MixLinear
            lora_kwargs = {}

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **lora_kwargs)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_mixout=False, lora_kwargs=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if lora_kwargs is not None:
            linear_layer = lora.Linear
        else:
            linear_layer = MixLinear if use_mixout else nn.Linear
            lora_kwargs = {}

        self.qkv = linear_layer(dim, dim * 3, bias=qkv_bias, **lora_kwargs)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_layer(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cur_depth=0,
                    is_tutel=True, moe_layers=None, lora_kwargs=None):
        
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.cur_layer = moe_layers[cur_depth] if moe_layers is not None else 'F'

        if self.cur_layer == 'M':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.0, proj_drop=drop, use_mixout=True, lora_kwargs=lora_kwargs)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.aux_loss = None
        self.is_moe_layer = False
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.is_tutel = is_tutel
        self.aux_loss_weights = 0.01
        if self.cur_layer == 'M':
            self.mlp = MixoutMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            return attn
        elif self.cur_layer == 'F' or self.cur_layer == 'M':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            raise Exception