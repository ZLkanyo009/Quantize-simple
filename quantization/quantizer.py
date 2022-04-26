
import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
    
    def update_qparams(self):
        raise NotImplementedError
    
    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams()
        quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale

        return fake_quant_tensor


class AsymmetricQuantizer_PerTensor(Quantizer):
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))

        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
                            )
    
    def update_qparams(self):
        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                     (self.quant_min - self.observer.min_val / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)