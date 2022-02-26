import torch
from .distance import sq_euclidean_distance
from .transforms import GreaterThan


class Noise(torch.nn.Module):
    def __init__(self, lower_bound=1e-5, constraint=None):
        super().__init__()
        self.raw_noise = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
        if constraint is None:
            self.constraint = GreaterThan(lower_bound)
        else:
            self.constraint = constraint

    @property
    def noise(self):
        return self.constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, x):
        with torch.no_grad():
            val = self.constraint.inverse_transform(torch.as_tensor(x))
            self.raw_noise.fill_(val)

    def forward(self, covar):
        diag = covar.diagonal()
        diag += self.noise
        return covar
