#!/usr/bin/env python3
"""
TODO
"""
import torch
import torch.nn.functional as F  # noqa: F401

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch import set_gpu_mode
from garage.torch.policies import GaussianMLPPolicy  # noqa: F401
from garage.torch.value_functions import GaussianMLPValueFunction  # noqa: F401
from garage.sampler import LocalSampler, RaySampler
from garage.trainer import Trainer

import gym
import safety_gym  # noqa: F401
import envs.custom_safety_envs  # noqa: F401
from safety_gym.envs.engine import Engine
from wrappers import RewardMasker, SafetyEnvStateAppender, Renderer
from algos import IrlTRPO  # noqa: F401
from buffers import SyntheticPreferenceCollector, LabelAnnealer  # noqa: F401
from reward_predictors import PrefMLP  # noqa: F401

import time
import os
import argparse
import json
import pickle


def robust_preferences(ctxt,
                       **kwargs,
                       ):

    kwargs['number_epochs'] += 1

    set_seed(kwargs['seed'])
    env = gym.make(kwargs['env_id'])

    if isinstance(env, Engine):
        config = env.config

        with open(os.path.join(ctxt.snapshot_dir, 'config.json'), 'w') \
                as outfile:
            json.dump(config, outfile)

    env.metadata['render.modes'] = ['rgb_array']

    env = RewardMasker(env)
    env = SafetyEnvStateAppender(env)
    env = Renderer(env, directory=os.path.join(ctxt.snapshot_dir, 'videos'))
    env = GymEnv(env, max_episode_length=kwargs['max_episode_length'])

    with open(os.path.join(ctxt.snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)
    policy = eval(kwargs['policy'])  # noqa: F841
    value_function = eval(kwargs['value_function'])  # noqa: F841
    label_scheduler = eval(kwargs['label_scheduler'])  # noqa: F841
    data_collector = eval(kwargs['data_collector'])  # noqa: F841
    reward_predictor = eval(kwargs['reward_predictor'])  # noqa: F841
    algo = eval(kwargs['algo'])

    if torch.cuda.is_available() and kwargs['use_gpu']:
        set_gpu_mode(True, gpu_id=kwargs['gpu_id'])
    else:
        set_gpu_mode(False)

    sampler = RaySampler if kwargs['ray'] else LocalSampler

    trainer.setup(
        algo=algo,
        env=env,
        sampler_cls=sampler,
        n_workers=kwargs['n_workers'],
    )

    trainer.train(n_epochs=kwargs['number_epochs'],
                  batch_size=kwargs['steps_per_epoch'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preference reward learning')
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--number_epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=8000)
    parser.add_argument('--snapshot_gap', type=int, default=100)
    parser.add_argument('--max_episode_length', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--ray', action='store_true', default=True)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)

    # These arguments will be evaluated as code
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--value_function', type=str, required=True)
    parser.add_argument('--label_scheduler', type=str, required=True)
    parser.add_argument('--data_collector', type=str, required=True)
    parser.add_argument('--reward_predictor', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)

    torch.set_num_threads(4)

    kwargs = vars(parser.parse_args())
    args = {k: v for k, v in kwargs.items() if v is not None}

    kwargs['name'] = (
        kwargs['name'] + '_' + time.ctime().replace(' ', '_')
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           kwargs['name'] + time.ctime().replace(' ', '_'))

    robust_preferences = wrap_experiment(
        robust_preferences,
        name=kwargs['name'],
        snapshot_gap=kwargs['snapshot_gap'],
        snapshot_mode='gapped_last',
        log_dir=log_dir,
    )
    robust_preferences(**kwargs)
