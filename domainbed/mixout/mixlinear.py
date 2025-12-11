##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.nn.modules.conv import _pair
from typing import Optional
from collections import OrderedDict
from .functional import mixout
from domainbed.models.drop import general_drop_out

class MixLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p)) 
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer, 
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p) 
    def __init__(self, 
                in_features:int, 
                out_features:int, 
                bias:bool=True, 
                p:float=0.0, activation:bool=False, scale_by_keep:bool=True, noise_lambda:float=0.0) -> None:

        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.p = p

        if self.p < 0 or self.p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {self.p}")

        self.target_w = Parameter(self.weight.new_empty(self.weight.size()))
        self.target_w.requires_grad = False

        self.activation = activation
        self.scale_by_keep = scale_by_keep
        self.noise_lambda = noise_lambda
        self.drop_mode = 'point'

        self.mask = None

    def make_noise(self, generator=None):
        shape = self.target_w.shape
        self.mask = self.target_w.new_empty(shape, device='cuda')
        self.mask.bernoulli_(1-self.p, generator=generator)
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return F.linear(input, mixout(self.weight, self.target_w, 
                                      self.p, self.training, scale_by_keep=self.scale_by_keep, noise_lambda=self.noise_lambda, drop_mode=self.drop_mode, mask=self.mask), self.bias)

    def extra_repr(self):
        type_ = 'drop' if self.target_w is None else 'mix'
        type_ += "out" 
        return f'{type_}={self.p}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class MixConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                padding: str = 0, dilation: int = 1, groups: int = 1, bias: bool = False,
                padding_mode: str = 'zeros', device=None, dtype=None, 
                p:float=0.0, drop_mode:str='filter', activation:bool=False, scale_by_keep:bool=True, noise_lambda:float=0.0) -> None:
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        assert drop_mode in ['point', 'filter', 'block'], "Mode must be either in ['point', 'filter', 'block']"

        self.target_w = Parameter(self.weight.new_empty(self.weight.size()))
        self.target_w.requires_grad = False

        if bias:
            self.target_b = Parameter(self.bias.new_empty(self.bias.size()))
            self.target_b.requires_grad = False
        else:
            self.target_b = None

        self.p = p
        self.drop_mode = drop_mode
        self.activation = activation
        self.scale_by_keep = scale_by_keep
        self.noise_lambda = noise_lambda

        self.mask = None

    def make_noise(self, generator=None):
        if self.drop_mode == 'filter':
            shape = (self.target_w.shape[0],) + (1,) * (self.target_w.ndim - 1)
        elif self.drop_mode == 'point': 
            shape = self.target_w.shape
        self.mask = self.target_w.new_empty(shape, device='cuda')
        self.mask.bernoulli_(1-self.p, generator=generator)

    def _mix_conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            mixout(weight, self.target_w, 
                                      self.p, self.training, scale_by_keep=self.scale_by_keep, noise_lambda=self.noise_lambda, drop_mode=self.drop_mode, mask=self.mask), bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, mixout(weight, self.target_w, 
                                      self.p, self.training, scale_by_keep=self.scale_by_keep, noise_lambda=self.noise_lambda, drop_mode=self.drop_mode, mask=self.mask), bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if self.activation:
            if self.training and self.p > 0.0: 
                return general_drop_out(x=self._conv_forward(input, self.weight, self.bias),
                                    fixed_x=self._conv_forward(input, self.target_w, self.target_b), p=self.p, scale_by_keep=self.scale_by_keep, drop_mode=self.drop_mode)
            else: return self._conv_forward(input, self.weight, self.bias)
        else:
            return self._mix_conv_forward(input, self.weight, self.bias)