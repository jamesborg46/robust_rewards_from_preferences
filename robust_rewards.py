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
from gym.envs.registration import register
import safety_gym
from safety_gym.envs.engine import Engine
from gym.wrappers import Monitor
from wrappers import RewardMasker, SafetyEnvStateAppender
from algos import PreferenceTRPO
from buffers import SyntheticComparisonCollector, HumanComparisonCollector, LabelAnnealer
from reward_predictors import MLPRewardPredictor, BNNRewardPredictor

import numpy as np

from datetime import datetime
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
                       segment_length=20,
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
                       **kwargs):


    """
    TODO DOCS
    """
    number_epochs = number_epochs + 1

    set_seed(seed)
    env = gym.make(env_id)
    config = env.config

    for k, v in kwargs.items():
        config[k] = v

    with open(os.path.join(ctxt.snapshot_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)

    env = Engine(config)
    env.metadata['render.modes'] = ['rgb_array']

    if monitor:
        # Defining this because pickle doesn't pickle lambda functions
        def video_callable(i):
            return (i % 100 == 0) and (i != 0)

        env = Monitor(env,
                      ctxt.snapshot_dir + '/monitoring',
                      force=True,
                      video_callable=video_callable)

    env = RewardMasker(env)
    env = SafetyEnvStateAppender(env)
    env = GymEnv(env, max_episode_length=max_episode_length)

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

        for _ in range(pre_train_labels):
            precollected.sample_comparison()

        precollected_comparisons = precollected._comparisons
    else:
        precollected_comparisons = []

    if comparison_collector_type == 'synthetic':
        comparison_collector = SyntheticComparisonCollector(
            4000,
            label_scheduler,
            segment_length=segment_length,
            precollected_comparisons=precollected_comparisons,
        )
    elif comparison_collector_type == 'human':
        comparison_collector = HumanComparisonCollector(
            20000,
            label_scheduler,
            env=env,
            experiment_name=name,
            segment_length=segment_length,
            precollected_comparisons=precollected_comparisons
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
                          center_adv=center_adv
                          )

    sampler_cls = LocalSampler if (monitor or local) else RaySampler

    trainer.setup(algo,
                  env,
                  sampler_cls=sampler_cls,
                  )

    trainer.train(n_epochs=number_epochs, batch_size=4000)


def register_envs():

    base_config = gym.make('Safexp-PointGoal1-v0').config
    new_config = {
        'robot_locations': [(0, -1.5)],
        'robot_rot': np.pi * (1/2),
        'goal_locations': [(0, 1.5)],
        'hazards_num': 3,
        'vases_num': 0,
        'observe_vases': False,
        'hazards_placements': None,
        'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 0)],
        'hazards_size': 0.4,
    }

    for k, v in new_config.items():
        assert k in base_config.keys(), 'BAD CONFIG'
        base_config[k] = v

    register(id='Safexp-PointGoalCustom0-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': base_config})

    base_config = gym.make('Safexp-PointGoal1-v0').config
    new_config = {
            'robot_locations': [(0, -1.5)],
            'robot_rot': np.pi * (1/2),
            'goal_locations': [(0, 1.5)],
            'hazards_num': 1,
            'vases_num': 1,
            'pillars_num': 1,
            'observe_vases': True,
            'observe_pillars': True,
            'constrain_vases': True,
            'constrain_pillars': True,
            'hazards_placements': None,
            'hazards_locations': [(0, 0)],
            'vases_placements': None,
            'vases_locations': [(-1.3, 0.9)],
            'pillars_placements': None,
            'pillars_locations': [(1.3, 0.9)],
            'hazards_size': 0.4,
            'pillars_size': 0.2,
            'vases_size': 0.2,
            'hazards_cost': 30,
            'vases_contact_cost': 30,
            'pillars_cost': 30,
    }

    for k, v in new_config.items():
        assert k in base_config.keys(), 'BAD CONFIG'
        base_config[k] = v

    register(id='Safexp-PointGoalThree0-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': base_config})

    base_config = gym.make('Safexp-PointGoal1-v0').config
    new_config = {
        'robot_locations': [(0, -1.5)],
        'robot_rot': np.pi * (1/2),
        'goal_locations': [(0, 0)],
        'hazards_num': 3,
        'vases_num': 0,
        'observe_vases': False,
        'hazards_placements': None,
        'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 1.5)],
        'hazards_size': 0.4,
        'hazards_cost': 30,
    }

    for k, v in new_config.items():
        assert k in base_config.keys(), 'BAD CONFIG'
        base_config[k] = v

    register(id='Safexp-PointGoalBehind0-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': base_config})


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
    parser.add_argument('--max_episode_length', type=int, required=False)
    parser.add_argument('--final_labels', type=int, required=False)
    parser.add_argument('--pre_train_labels', type=int, required=False)
    parser.add_argument('--monitor', action='store_true', required=False)
    parser.add_argument('--local', action='store_true', required=False)
    parser.add_argument('--use_gt_rewards', action='store_true', required=False)
    parser.add_argument('--discount', type=float, required=False)
    parser.add_argument('--val_opt_its', type=int, required=False)
    parser.add_argument('--val_opt_lr', type=float, required=False)
    parser.add_argument('--reward_opt_its', type=int, required=False)
    parser.add_argument('--reward_opt_lr', type=float, required=False)
    parser.add_argument('--center_adv', action='store_true', default=False)

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    config = {
        'continue_goal': False,
        'reward_includes_cost': True,
        'hazards_cost': 30.,
        'constrain_indicator': False,
        'reward_goal': 0,
    }

    args['name'] = (
        args['name'] + '_' + datetime.now().strftime("%m%d%Y_%H%M%S")
    )

    kwargs = {**args, **config}

    register_envs()

    robust_preferences = wrap_experiment(
        robust_preferences,
        name=args['name'],
        snapshot_gap=25,
        snapshot_mode='gap'
    )
    robust_preferences(**kwargs)
