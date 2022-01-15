import torch
from .distance import sq_euclidean_distance
from .transforms import log_exp


def rbf(x, y, lengthscale):
    """
    RBF kernel.
    """
    lengthscale = log_exp(lengthscale)

    dist = sq_euclidean_distance(x / lengthscale, y / lengthscale)
    return dist.div_(-2).exp_()


def rbf_init(model, k_i):
    model.register_parameter(
        f"raw_lengthscale{k_i}", torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
    )
