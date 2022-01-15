import torch


def sq_euclidean_distance(x, y):
    for inp in [x, y]:
        assert len(inp.shape) == 1 or inp.shape[1] == 1
    return torch.pow(x.reshape(-1, 1) - y.reshape(1, -1), 2)


def hamming_distance(x, y):
    pass
