
import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()

    def update_range(self, input):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        self.update_range(input)

        return input

class MinMaxObserver_PerTensor(ObserverBase):
    def __init__(self, out_channels):
        super(MinMaxObserver_PerTensor, self).__init__()
        self.out_channels = out_channels
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, input):
        min_val = torch.min(input)
        max_val = torch.max(input)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

class EMAMinMaxObserver_PerTensor(ObserverBase):
    def __init__(self, out_channels, momentum=0.1):
        super(EMAMinMaxObserver_PerTensor, self).__init__()
        self.momentum = momentum
        self.out_channels = out_channels
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, input):
        min_val_cur = torch.min(input)
        max_val_cur = torch.max(input)

        min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
