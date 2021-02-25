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
from modules import MLPRewardPredictor, BNNRewardPredictor
from dowel import tabular, logger

import numpy as np

from datetime import datetime
import os
import argparse
import json
import pickle

DEVICE = 'cuda'

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
        hidden_sizes=(126, 126),
        layer_normalization=True,
        hidden_nonlinearity=F.relu,
    ).to(DEVICE)

    optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-3, weight_decay=1e-2)),
        # torch.optim.Adam,
        reward_predictor,
        minibatch_size=256,
        max_optimization_epochs=1
    )

#     reward_predictor = BNNRewardPredictor(
#         env_spec=env.spec,
#     )

    if not use_total_ordering:
        left_segs, right_segs, preferences = get_data_from_pairwise(comparison_collector)
    else:
        total_ordering, sorted_idx = get_total_ordering(comparison_collector._segments)
        # segs, ranks = get_totall_ordered_ranked_data(total_ordering, sorted_idx)
        # segs = (segs - segs.mean(dim=0)) / segs.std(dim=0)

    for i in range(epochs):
        if use_total_ordering:
            left_segs, right_segs, preferences = get_totally_ordered_data(
                total_ordering,
                5,
            )
            # pass

        accuracies = []
        for _left_segs, _right_segs, _prefs in optimizer.get_minibatch(
                left_segs, right_segs, preferences):
            loss, accuracy = train_reward_predictor(reward_predictor,
                                   _left_segs,
                                   _right_segs,
                                   _prefs,
                                   optimizer,)
            accuracies.append(accuracy.cpu().item())

        with tabular.prefix('/RewardExperiments'):
            tabular.record('/TrainAccuracy', np.mean(accuracies))

#         for _segs, _ranks in optimizer.get_minibatch(segs, ranks):
#             train_ranked_reward_predictor(reward_predictor,
#                                    _segs,
#                                    _ranks,
#                                    optimizer,)

        validate_reward_predictor(reward_predictor,
                                  sorted_idx,
                                  comparison_collector,
                                  test_comparison_collector,
                                  left_segs,
                                  right_segs,
                                  preferences)

        left_segs, right_segs, preferences = get_totally_ordered_data(
            total_ordering,
            5,
        )

        test_accuracies = []
        with torch.no_grad():
            for _left_segs, _right_segs, _prefs in optimizer.get_minibatch(
                    left_segs, right_segs, preferences):
                loss, accuracy = reward_predictor.compute_preference_loss(
                        _left_segs,
                        _right_segs,
                        _prefs,
                        device=DEVICE
                        # dataset_size=len(self.comparison_collector.labeled_decisive_comparisons)
                     )
                test_accuracies.append(accuracy.cpu().item())

        with tabular.prefix('/RewardExperiments'):
            tabular.record('/TestAccuracy', np.mean(test_accuracies))

        logger.log(tabular)
        logger.dump_all(step=i)


def train_ranked_reward_predictor(reward_predictor,
                           segs,
                           ranks,
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
    loss = reward_predictor.propagate_ranking_loss(
        segs,
        ranks,
        # dataset_size=len(self.comparison_collector.labeled_decisive_comparisons)
     )
    optimizer.step()

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/Loss', loss)
    return loss


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
    loss, accuracy = reward_predictor.propagate_preference_loss(
        left_segs,
        right_segs,
        prefs,
        device=DEVICE
        # dataset_size=len(self.comparison_collector.labeled_decisive_comparisons)
     )
    optimizer.step()

    return loss, accuracy


def validate_reward_predictor(reward_predictor,
                              segments_sorted_idx,
                              train_comparison_collector,
                              test_comparison_collector,
                              left_segs,
                              right_segs,
                              preferences):

    num_train_segments = len(train_comparison_collector._segments)

    segs = []
    gt_rewards = []
    gt_reward_sums = []
    for segment in train_comparison_collector._segments:
        segs.append(segment['observations'][0])
        gt_rewards.append(segment['gt_rewards'][0])
        # gt_reward_sums.append(segment['gt_rewards'].sum())

    with torch.no_grad():
        # segs = torch.tensor(np.concatenate(segs)).type(torch.float32).to(DEVICE)
        segs = torch.tensor(np.stack(segs)).type(torch.float32).to(DEVICE)
        pred_rewards = reward_predictor(segs).cpu()

    segment_correlation = corrcoef(pred_rewards.flatten().numpy(),
                                   np.array(gt_rewards).flatten())

    # pred_seg_reward_sums = pred_rewards.reshape(num_train_segments,
    #                                             5).sum(dim=1).numpy()

    # pred_rewards_sorted_idx = np.argsort(pred_seg_reward_sums)
    # gt_reward_sums_sorted_idx = np.argsort(gt_reward_sums)
    pred_rewards_sorted_idx = np.argsort(pred_rewards.flatten().numpy())
    gt_reward_sums_sorted_idx = np.argsort(np.array(gt_rewards).flatten())

    with tabular.prefix('/RewardExperiments'):
        tabular.record('/SegmentCorrelation', segment_correlation)
        tabular.record('/AvgFootruleDistance',
                       np.sum(np.abs(pred_rewards_sorted_idx -
                                     gt_reward_sums_sorted_idx)) /
                       num_train_segments)
        tabular.record('/AvgFootruleBaseline',
                       np.sum(np.abs(np.random.permutation(num_train_segments) -
                                     gt_reward_sums_sorted_idx)) /
                       num_train_segments)


    path_corrs = []
    std_devs = []
    means = []
    for path in train_comparison_collector._paths:

        with torch.no_grad():
            path_tensor = torch.tensor(path['observations']).type(torch.float32).to(DEVICE)
            pred_rewards = reward_predictor(path_tensor).cpu()

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
    ).type(torch.float32).to(DEVICE)

    right_segs = torch.tensor(
        [comp['right']['observations'] for comp in labeled_comparisons]
    ).type(torch.float32).to(DEVICE)

    preferences = torch.tensor(
        [comp['label'] for comp in labeled_comparisons]
    ).type(torch.long).to(DEVICE)

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

    all_left_segs = torch.cat(all_left_segs).to(DEVICE)
    all_right_segs = torch.cat(all_right_segs).to(DEVICE)
    all_prefs = torch.cat(all_prefs).to(DEVICE)

    return all_left_segs, all_right_segs, all_prefs


def get_totall_ordered_ranked_data(ordered_segments, sorted_idx):
    segs = torch.tensor([seg['observations'][0] for seg in ordered_segments]).type(torch.float32).to(DEVICE)
    ranks = torch.tensor(sorted_idx).type(torch.float32).to(DEVICE)
    return segs, ranks


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
