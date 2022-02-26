import torch


class LogExp(torch.nn.Module):
    def transform(self, x):
        return torch.log(1 + torch.exp(x))

    def inverse_transform(self, x):
        return torch.log(torch.exp(x) - 1)


class Exp(torch.nn.Module):
    def transform(self, x):
        return torch.exp(x)

    def inverse_transform(self, x):
        return torch.log(x)


class GreaterThan(torch.nn.Module):
    def __init__(self, lower_bound, base_transform=LogExp()):
        super().__init__()
        self.lower_bound = lower_bound
        self.base_transform = base_transform

    def transform(self, x):
        return self.base_transform.transform(x) + self.lower_bound

    def inverse_transform(self, x):
        return self.base_transform.inverse_transform(x - self.lower_bound)
