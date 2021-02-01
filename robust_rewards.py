#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
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

import gym
import safety_gym
import envs.custom_safety_envs
from safety_gym.envs.engine import Engine
from gym.wrappers import Monitor
from wrappers import RewardMasker, SafetyEnvStateAppender
from algos import PreferenceTRPO
from buffers import SyntheticComparisonCollector, \
    HumanComparisonCollector, LabelAnnealer
from reward_predictors import MLPRewardPredictor

import time
import os
import argparse
import json
import pickle


def robust_preferences(ctxt=None,
                       seed=1,
                       name='EXP',
                       env_id='Safexp-PointGoal0-v0',
                       comparison_collector_type='synthetic',
                       number_epochs=1000,
                       segment_length=5,
                       max_episode_length=1000,
                       final_labels=1000,
                       pre_train_labels=400,
                       monitor=False,
                       local=False,
                       use_gt_rewards=False,
                       discount=0.99,
                       val_opt_its=100,
                       val_opt_lr=1e-3,
                       reward_opt_its=20,
                       reward_opt_lr=1e-3,
                       center_adv=True,
                       precollected_trajectories=None,
                       totally_ordered=False,
                       n_workers=8,
                       **kwargs):

    """
    TODO DOCS
    """
    number_epochs = number_epochs + 1

    set_seed(seed)
    env = gym.make(env_id)

    if isinstance(env, Engine):
        config = env.config

        with open(os.path.join(ctxt.snapshot_dir, 'config.json'), 'w') \
                as outfile:
            json.dump(config, outfile)

    env.metadata['render.modes'] = ['rgb_array']

    env = RewardMasker(env)
    env = SafetyEnvStateAppender(env)
    env = GymEnv(env, max_episode_length=max_episode_length)

    with open(os.path.join(ctxt.snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    reward_predictor = MLPRewardPredictor(
        env_spec=env.spec,
        hidden_sizes=(16, 16),
        hidden_nonlinearity=F.relu,
    )

#     reward_predictor = BNNRewardPredictor(
#         env_spec=env.spec,
#     )

    label_scheduler = LabelAnnealer(final_timesteps=number_epochs,
                                    final_labels=final_labels,
                                    pretrain_labels=pre_train_labels)

    if precollected_trajectories is not None:
        with open(precollected_trajectories, 'rb') as f:
            precollected = pickle.load(f)

        obs_space = env.observation_space.shape[0]

        for i in range(len(precollected._paths)):
            precollected._paths[i]['observations'] = (
                precollected._paths[i]['observations'][:, -obs_space:]
            )

        if (not hasattr(precollected, '_segments')) or \
                (len(precollected._segments) / 2) != len(precollected._comparisons):
            precollected._segments = SyntheticComparisonCollector.comps_to_segs(
                precollected._comparisons
            )

        precollected.segment_length = segment_length
        for _ in range(pre_train_labels):
            precollected.sample_comparison()

        precollected_segments = precollected._segments
        precollected_comparisons = precollected._comparisons
    else:
        precollected_comparisons = []
        precollected_segments = []

    if comparison_collector_type == 'synthetic':
        comparison_collector = SyntheticComparisonCollector(
            4000,
            label_scheduler,
            segment_length=segment_length,
            precollected_comparisons=precollected_comparisons,
            precollected_segments=precollected_segments,
        )
    elif comparison_collector_type == 'human':
        comparison_collector = HumanComparisonCollector(
            20000,
            label_scheduler,
            env=env,
            experiment_name=name,
            segment_length=segment_length,
            precollected_comparisons=precollected_comparisons,
            precollected_segments=precollected_segments,
        )

    test_comparison_collector = SyntheticComparisonCollector(
        40000,
        label_scheduler,
        collect_callable=lambda i: i % 10 == 0
    )

    algo = PreferenceTRPO(env_spec=env.spec,
                          policy=policy,
                          value_function=value_function,
                          reward_predictor=reward_predictor,
                          comparison_collector=comparison_collector,
                          test_comparison_collector=test_comparison_collector,
                          use_gt_rewards=use_gt_rewards,
                          discount=discount,
                          val_opt_its=val_opt_its,
                          val_opt_lr=val_opt_lr,
                          reward_opt_its=reward_opt_its,
                          reward_opt_lr=reward_opt_lr,
                          center_adv=center_adv,
                          totally_ordered=totally_ordered,
                          )

    sampler_cls = LocalSampler if (monitor or local) else RaySampler

    trainer.setup(algo,
                  env,
                  sampler_cls=sampler_cls,
                  n_workers=n_workers,
                  )

    trainer.train(n_epochs=number_epochs, batch_size=4000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preference reward learning')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=False)
    parser.add_argument('--comparison_collector_type', type=str,
                        required=False)
    parser.add_argument('--precollected_trajectories', type=str,
                        required=False)
    parser.add_argument('--number_epochs', type=int, required=False)
    parser.add_argument('--segment_length', type=int, required=False)
    parser.add_argument('--snapshot_gap', type=int, default=100)
    parser.add_argument('--max_episode_length', type=int, required=False)
    parser.add_argument('--final_labels', type=int, required=False)
    parser.add_argument('--pre_train_labels', type=int, required=False)
    parser.add_argument('--monitor', action='store_true', required=False)
    parser.add_argument('--totally_ordered',
                        action='store_true', required=False)
    parser.add_argument('--local', action='store_true', required=False)
    parser.add_argument('--use_gt_rewards',
                        action='store_true', required=False)
    parser.add_argument('--discount', type=float, required=False)
    parser.add_argument('--val_opt_its', type=int, required=False)
    parser.add_argument('--val_opt_lr', type=float, required=False)
    parser.add_argument('--reward_opt_its', type=int, required=False)
    parser.add_argument('--reward_opt_lr', type=float, required=False)
    parser.add_argument('--center_adv', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=8)

    torch.set_num_threads(4)

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    args['name'] = (
        args['name'] + '_' + time.ctime().replace(' ', '_')
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           args['name'] + time.ctime().replace(' ', '_'))

    robust_preferences = wrap_experiment(
        robust_preferences,
        name=args['name'],
        snapshot_gap=args['snapshot_gap'],
        snapshot_mode='gapped_last',
        log_dir=log_dir,
    )
    robust_preferences(**args)
