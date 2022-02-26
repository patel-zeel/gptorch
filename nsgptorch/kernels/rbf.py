import torch
from ..distance import sq_euclidean_distance
from ..transforms import LogExp


class RBF(torch.nn.Module):
    def __init__(self, constraint=None, active_dim: int = None):
        super().__init__()
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
        if constraint is None:
            self.constraint = LogExp()
        else:
            self.constraint = constraint
        self.active_dim = active_dim

    @property
    def lengthscale(self):
        return self.constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, x):
        with torch.no_grad():
            val = self.constraint.inverse_transform(torch.as_tensor(x))
            self.raw_lengthscale.fill_(val)

    def forward(self, x1, x2):
        assert self.active_dim is not None, "active_dim must be specified"
        x1 = x1[:, self.active_dim]
        x2 = x2[:, self.active_dim]

        diff = sq_euclidean_distance(x1 / self.lengthscale, x2 / self.lengthscale)
        return diff.div_(-2).exp_()

    def is_stationary(self):
        return True
