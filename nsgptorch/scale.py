import torch
from .transforms import LogExp


class Scale(torch.nn.Module):
    def __init__(self, constraint=None):
        super().__init__()
        self.raw_outputscale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
        if constraint is None:
            self.constraint = LogExp()
        else:
            self.constraint = constraint

    @property
    def outputscale(self):
        return self.constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, x):
        with torch.no_grad():
            val = self.constraint.inverse_transform(torch.as_tensor(x))
            self.raw_outputscale.fill_(val)

    def forward(self, res):
        return self.outputscale * res
