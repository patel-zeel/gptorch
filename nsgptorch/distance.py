import torch


def sq_euclidean_distance(x, y):
    """
    x and y must be 1D arrays of the same length with shape (N,).
    """
    return torch.pow(x.reshape(-1, 1) - y.reshape(1, -1), 2)


def hamming_distance(x, y):
    pass
