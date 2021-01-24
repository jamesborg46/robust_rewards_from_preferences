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
