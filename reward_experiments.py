import torch
import torch.nn.functional as F

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
# from garage.torch.modules import MLPModule
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer
from garage.torch.optimizers import OptimizerWrapper

import gym
from gym.envs.registration import register
import safety_gym
from safety_gym.envs.engine import Engine
from gym.wrappers import Monitor
from wrappers import RewardMasker, SafetyEnvStateAppender
from algos import PreferenceTRPO
from buffers import SyntheticComparisonCollector, HumanComparisonCollector
from reward_predictors import MLPRewardPredictor, BNNRewardPredictor
from dowel import tabular, logger

import numpy as np

from datetime import datetime
import os
import argparse
import json
import pickle


HOME = (
    "/home/mil/james/safety_experiments/"
    "robust_rewards_from_preferences/data/local/experiment"
)


def main(ctxt,
         exp,
         test_exp,
         comparisons,
         test_comparisons,
         epochs=1000,
         use_total_ordering=False,
         ):

    with open(os.path.join(HOME, exp, 'config.json'), 'rb') as f:
        config = json.load(f)

    with open(os.path.join(HOME, exp, comparisons), 'rb') as f:
        comparison_collector = pickle.load(f)

    with open(os.path.join(HOME, test_exp, test_comparisons), 'rb') as f:
        test_comparison_collector = pickle.load(f)

    if not hasattr(comparison_collector, '_segments'):
        comparison_collector._segments = (
            SyntheticComparisonCollector.comps_to_segs(
                comparison_collector._comparisons
            )
        )

    if not hasattr(test_comparison_collector, '_segments'):
        test_comparison_collector._segments = (
            SyntheticComparisonCollector.comps_to_segs(
                test_comparison_collector._comparisons
            )
        )

    env = Engine(config)

    env = RewardMasker(env)
    env = SafetyEnvStateAppender(env)
    env = GymEnv(env, max_episode_length=1000)


    reward_predictor = MLPRewardPredictor(
        env_spec=env.spec,
        hidden_sizes=(16, 16),
        hidden_nonlinearity=F.relu,
    )

    optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-3)),
        # torch.optim.Adam,
        reward_predictor,
        max_optimization_epochs=1
    )

#     reward_predictor = BNNRewardPredictor(
#         env_spec=env.spec,
#     )

    if not use_total_ordering:
        left_segs, right_segs, preferences = get_data_from_pairwise(comparison_collector)
    else:
        total_ordering, sorted_idx = get_total_ordering(comparison_collector._segments)

    for i in range(epochs):
        if use_total_ordering:
            left_segs, right_segs, preferences = get_totally_ordered_data(
                total_ordering,
                1,
            )

        for left_segs, right_segs, prefs in optimizer.get_minibatch(
                left_segs, right_segs, preferences):
            train_reward_predictor(reward_predictor,
                                   left_segs,
                                   right_segs,
                                   prefs,
                                   optimizer,)

        validate_reward_predictor(reward_predictor,
                                  sorted_idx,
                                  comparison_collector,
                                  test_comparison_collector)
        logger.log(tabular)
        logger.dump_all(step=i)


def train_reward_predictor(reward_predictor,
                           left_segs,
                           right_segs,
                           prefs,
                           optimizer):
    r"""Train the reward predictor.
    Args:
        left_segs (torch.Tensor): Observation from the environment
            with shape :math:`(N, O*)`.
        right_segs (torch.Tensor): Observation from the environment
            with shape :math:`(N, O*)`.
        prefs (torch.Tensor): Acquired returns
            with shape :math:`(N, )`.
    Returns:
        torch.Tensor: Calculated mean scalar value of value function loss
            (float).
    """
    optimizer.zero_grad()
    loss = reward_predictor.propagate_preference_loss(
        left_segs,
        right_segs,
        prefs,
        # dataset_size=len(self.comparison_collector.labeled_decisive_comparisons)
     )
    optimizer.step()

    return loss


def validate_reward_predictor(reward_predictor,
                              segments_sorted_idx,
                              train_comparison_collector,
                              test_comparison_collector):

    segs = []
    gt_rewards = []
    for segment in train_comparison_collector._segments:
        segs.append(segment['observations'])
        gt_rewards.append(segment['gt_rewards'])

    with torch.no_grad():
        segs = torch.tensor(np.concatenate(segs)).type(torch.float32)
        pred_rewards = reward_predictor(segs)

    breakpoint()

    segment_correlation = corrcoef(pred_rewards.flatten().numpy(),
                                   np.array(gt_rewards).flatten())

    pred_rewards_sorted_idx = np.argsort(pred_rewards)

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/SegmentCorrelation', segment_correlation)
        # tabular.record('/FootruleDistance',
        #                np.sum(np.abs(segments_sorted_idx -
        #                              pred_rewards_sorted_idx)))


    path_corrs = []
    std_devs = []
    means = []
    for path in train_comparison_collector._paths:

        with torch.no_grad():
            path_tensor = torch.tensor(path['observations']).type(torch.float32)
            pred_rewards = reward_predictor(path_tensor)

        path_corrs.append(corrcoef(pred_rewards.flatten().numpy(),
                                   path['gt_rewards'].flatten()))
        std_devs.append(pred_rewards.flatten().std().item())
        means.append(pred_rewards.flatten().mean().item())

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/AvgPathhCorrelation', np.mean(path_corrs))

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/AvgStdRewPath', np.mean(std_devs))

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/MeanRewPath', np.mean(means))

def get_data_from_pairwise(comparison_collector):
    labeled_comparisons = comparison_collector.labeled_decisive_comparisons

    left_segs = torch.tensor(
        [comp['left']['observations'] for comp in labeled_comparisons]
    ).type(torch.float32)

    right_segs = torch.tensor(
        [comp['right']['observations'] for comp in labeled_comparisons]
    ).type(torch.float32)

    preferences = torch.tensor(
        [comp['label'] for comp in labeled_comparisons]
    ).type(torch.long)

    return left_segs, right_segs, preferences


# def get_totally_ordered_data(segments, epochs=20):
#     N = len(segments)
#     permuted_idxs = np.random.permutation(N)
#     left_idxs = permuted_idxs[:N//2]
#     right_idxs = permuted_idxs[N//2:]
#     left_segs = torch.tensor(
#         [seg['observations'] for seg in segments[left_idxs]]
#     ).type(torch.float32)
#     right_segs = torch.tensor(
#         [seg['observations'] for seg in segments[right_idxs]]
#     ).type(torch.float32)
#     prefs = torch.tensor(right_idxs > left_idxs).type(torch.long)
#     return left_segs, right_segs, prefs

def get_totally_ordered_data(ordered_segments, epochs=20):
    N = len(ordered_segments)

    all_left_segs = []
    all_right_segs = []
    all_prefs = []
    for _ in range(epochs):
        permuted_idxs = np.random.permutation(N)

        left_idxs = permuted_idxs[:N//2]
        right_idxs = permuted_idxs[N//2:]

        left_segs = torch.tensor(
            [seg['observations'] for seg in ordered_segments[left_idxs]]
        ).type(torch.float32)

        all_left_segs.append(left_segs)

        right_segs = torch.tensor(
            [seg['observations'] for seg in ordered_segments[right_idxs]]
        ).type(torch.float32)

        all_right_segs.append(right_segs)

        prefs = torch.tensor(right_idxs > left_idxs).type(torch.long)
        all_prefs.append(prefs)

    all_left_segs = torch.cat(all_left_segs)
    all_right_segs = torch.cat(all_right_segs)
    all_prefs = torch.cat(all_prefs)

    return left_segs, right_segs, prefs



def get_total_ordering(segments):
    segment_total_rewards = []
    for segment in segments:
        segment_total_rewards.append(segment['gt_rewards'].sum())
    sorted_idx = np.argsort(segment_total_rewards)
    return np.array(segments)[sorted_idx], sorted_idx


def corrcoef(dist_a, dist_b):
    """Returns a scalar between 1.0 and -1.0. 0.0 is no correlation. 1.0 is perfect correlation"""
    dist_a = np.copy(dist_a)  # Prevent np.corrcoef from blowing up on data with 0 variance
    dist_b = np.copy(dist_b)
    dist_a[0] += 1e-12
    dist_b[0] += 1e-12
    return np.corrcoef(dist_a, dist_b)[0, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='reward experiments')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--test_exp', type=str, required=True)
    parser.add_argument('--comparisons', type=str, required=True)
    parser.add_argument('--test_comparisons', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--use_total_ordering', action='store_true', default=False)

    args = parser.parse_args()

    main = wrap_experiment(
        main,
        name=args.name,
    )

    main(
        exp=args.exp,
        test_exp=args.test_exp,
        comparisons=args.comparisons,
        test_comparisons=args.test_comparisons,
        epochs=args.epochs,
        use_total_ordering=args.use_total_ordering,
    )
