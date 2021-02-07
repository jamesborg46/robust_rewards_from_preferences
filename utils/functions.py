import numpy as np
import torch
import akro


def split_flattened(space, values):
    spaces = space.spaces

    if isinstance(values, np.ndarray):
        dims = np.array([s.flat_dim for s in spaces.values()])
        flat_x = np.split(values, np.cumsum(dims)[:-1], axis=-1)
        return dict(zip(spaces.keys(), flat_x))
    elif isinstance(values, torch.Tensor):
        dims = [s.flat_dim for s in spaces.values()]
        flat_x = torch.split(values, dims, dim=-1)
        return dict(zip(spaces.keys(), flat_x))


def one_hot_to_int(arr):
    if isinstance(arr, np.ndarray):
        return np.argmax(arr, axis=-1)
    elif isinstance(arr, torch.Tensor):
        return torch.argmax(arr, axis=-1)


def int_to_one_hot(arr, n):
    if isinstance(arr, np.ndarray):
        return np.eye(n)[arr]
    elif isinstance(arr, np.ndarray):
        return torch.eye(n)[arr]


def corrcoef(dist_a, dist_b):
    """Returns a scalar between 1.0 and -1.0. 0.0 is no correlation. 1.0 is perfect correlation"""
    dist_a = np.copy(dist_a)  # Prevent np.corrcoef from blowing up on data with 0 variance
    dist_b = np.copy(dist_b)
    dist_a[0] += 1e-12
    dist_b[0] += 1e-12
    return np.corrcoef(dist_a, dist_b)[0, 1]
