
from . import quantizer

import torch
import torch.nn as nn
import torch.nn.functional as F

class QConv2d(nn.Conv2d):
    def __init__(self, ptq, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', bit=8,
                 sign=False, **kwargs):

        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        
        self.weight_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                        observer = observer.MinMaxObserver_PerTensor(None),
                                                                        ptq = ptq,
                                                                        sign = self.sign)
        self.input_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                       observer = observer.EMAMinMaxObserver_PerTensor(None),
                                                                       ptq = ptq,
                                                                       sign = self.sign)

    def forward(self, input):
        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.conv2d(
            input, weight_quant, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return output
