import time

import numpy as np
import torch

from dowel import tabular
from garage import StepType
from garage.np import discount_cumsum


def profile(label, func, *args, prefix='Profile', **kwargs):
    """This function is a wrapper to profile the given function

    Parameters
    ----------
    func : TODO
    args : TODO
    label : TODO

    Returns
    -------
    TODO

    """
    start = time.time()
    ret = func(*args, **kwargs)
    runtime = time.time() - start
    with tabular.prefix(prefix + '/'):
        tabular.record(label+'Time', runtime)
    return ret


def log_gt_performance(itr, batch, discount, prefix='GTEvaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    corrs = []
    for eps in batch.split():
        returns.append(discount_cumsum(eps.env_infos['gt_reward'], discount))
        undiscounted_returns.append(sum(eps.env_infos['gt_reward']))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))
        corrs.append(corrcoef(eps.rewards, eps.env_infos['gt_reward']))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))

        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        tabular.record('AverageEpisodeRewardCorrelation',
                       np.mean(corrs))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns


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


# def update_remote_agent_device(policy, device='cpu'):
#     params = policy.get_param_values()
#     if 'inner_params' in params.keys():
#         params['inner_params'] = OrderedDict(
#             [(k, v.to(device)) for k, v in params['inner_params'].items()]
#         )
#     else:
#         params = OrderedDict(
#             [(k, v.to(device)) for k, v in params.items()]
#         )
#     return params
