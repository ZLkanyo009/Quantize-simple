
from . import quantizer
from . import observer

import torch
import torch.nn as nn
import torch.nn.functional as F

class QLinear(nn.Linear):
    def __init__(self, ptq, in_features, out_features, bias=True, bit=8,
                 sign=True, **kwargs):

        super(QLinear, self).__init__(in_features, out_features, bias)

        self.weight_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                        observer = observer.MinMaxObserver_PerTensor(None),
                                                                        ptq = ptq,
                                                                        sign = sign)
        self.input_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
                                                                       observer = observer.EMAMinMaxObserver_PerTensor(None),
                                                                       ptq = ptq,
                                                                       sign = sign)

    def forward(self, input):
        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.linear(input, weight_quant, self.bias)
        return output
