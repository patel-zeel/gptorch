import torch


def log_exp(param):
    return torch.log(1 + torch.exp(param))
    # return torch.nn.Softplus()(param)
