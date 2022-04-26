from . import quantizer
from . import observer

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class QConv2dBn(nn.Conv2d):
    def __init__(self, ptq, total_steps, bn, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', bit=8,
                 sign=True, **kwargs):

        super(QConv2dBn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        
        self.register_buffer(
            "running_mean", copy.deepcopy(bn.running_mean)
        )
        self.register_buffer(
            "running_var", copy.deepcopy(bn.running_var)
        )
        self.register_buffer('steps', torch.zeros(1, device = bn.running_mean.device))

        self.gamma = copy.deepcopy(bn.weight)
        self.beta = copy.deepcopy(bn.bias)
        self.momentum = bn.momentum
        self.eps = bn.eps
        self.freeze_steps = 0.9 * total_steps

        self.weight_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                        observer = observer.MinMaxObserver_PerTensor(None),
                                                                        ptq = ptq,
                                                                        sign = sign)
        self.input_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                       observer = observer.EMAMinMaxObserver_PerTensor(None),
                                                                       ptq = ptq,
                                                                       sign = sign)

    def forward(self, input):
        if self.training:
            self.steps += 1
            # 先做普通卷积得到A，以取得BN参数
            output = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            # 更新BN统计参数（batch和running）
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            
            with torch.no_grad():
                running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.running_mean.copy_(running_mean)
                self.running_var.copy_(running_var)
            
            if self.steps < self.freeze_steps:
                #bn bias融合
                if self.bias is not None:
                    bias_fused = (
                        self.beta
                        + (self.bias - batch_mean)
                        * (self.gamma / torch.sqrt(batch_var + self.eps))
                    ).reshape(-1)
                else:
                    bias_fused = (
                        self.beta
                        - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps))
                    ).reshape(-1)
                
                weight_fused = self.weight * (
                            self.gamma / torch.sqrt(batch_var + self.eps)
                        ).reshape(-1, 1, 1, 1)
            else:
                print("freeze bn==============")
                if self.bias is not None:
                    bias_fused = (
                        self.beta
                        + (self.bias - self.running_mean)
                        * (self.gamma / torch.sqrt(self.running_var + self.eps))
                    ).reshape(-1)
                else:
                    bias_fused = (
                        self.beta
                        - self.running_mean
                        * (self.gamma / torch.sqrt(self.running_var + self.eps))
                    ).reshape(-1)

                weight_fused = self.weight * (
                    self.gamma / torch.sqrt(self.running_var + self.eps)
                ).reshape(-1, 1, 1, 1)
        else:
            if self.bias is not None:
                bias_fused = (
                    self.beta
                    + (self.bias - self.running_mean)
                    * (self.gamma / torch.sqrt(self.running_var + self.eps))
                ).reshape(-1)
            else:
                bias_fused = (
                    self.beta
                    - self.running_mean
                    * (self.gamma / torch.sqrt(self.running_var + self.eps))
                ).reshape(-1)

            weight_fused = self.weight * (
                self.gamma / torch.sqrt(self.running_var + self.eps)
            ).reshape(-1, 1, 1, 1)


        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(weight_fused)

        output = F.conv2d(
            input, weight_quant, bias_fused, self.stride, self.padding,
            self.dilation, self.groups)
        return output