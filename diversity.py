import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import gym
import safety_gym
from safety_gym.envs.engine import Engine
import envs.custom_safety_envs

from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler, RaySampler, DefaultWorker
from garage.torch import set_gpu_mode
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from garage.torch.modules import MLPModule

from wrappers import DiversityWrapper, SafetyEnvStateAppender, Renderer
from algos import DIAYN

import json
import time
import argparse
import os

import pickle

import dowel
from dowel import logger


def diversity_is_all_you_need(ctxt=None,
                              seed=1,
                              name='EXP',
                              number_skills=10,
                              number_epochs=1001,
                              max_episode_length=1000,
                              render_freq=200,
                              alpha=0.1,
                              n_workers=8,
                              env_id='Safexp-PointGoal0-v0',
                              batch_size=4000,
                              ray=False,
                              use_gpu=False,
                              gpu_id=0,
                              **kwargs,
                              ):

    set_seed(seed)
    env = gym.make(env_id)

    if isinstance(env, Engine):
        config = env.config

        with open(os.path.join(ctxt.snapshot_dir, 'config.json'), 'w') \
                as outfile:
            json.dump(config, outfile)

    env.metadata['render.modes'] = ['rgb_array']

    env = DiversityWrapper(env, number_skills=number_skills)
    env = SafetyEnvStateAppender(env)
    env = Renderer(env, os.path.join(ctxt.snapshot_dir, 'videos'))
    env = GymEnv(env, max_episode_length=max_episode_length)

    with open(os.path.join(ctxt.snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)

    skill_discriminator = MLPModule(
        input_dim=env.original_obs_space.shape[0],
        output_dim=number_skills,
        hidden_sizes=(256, 256),
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=nn.LogSoftmax,
    )

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sac = DIAYN(env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                discriminator=skill_discriminator,
                number_skills=number_skills,
                render_freq=render_freq,
                gradient_steps_per_itr=300,
                discriminator_gradient_steps_per_itr=300,
                max_episode_length_eval=1000,
                replay_buffer=replay_buffer,
                min_buffer_size=1e4,
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=256,
                num_evaluation_episodes=number_skills,
                reward_scale=1.,
                fixed_alpha=alpha,
                steps_per_epoch=1,
                use_deterministic_evaluation=False)

    if torch.cuda.is_available() and use_gpu:
        set_gpu_mode(True, gpu_id=gpu_id)
    else:
        set_gpu_mode(False)

    sampler = RaySampler if ray else LocalSampler

    trainer.setup(
        algo=sac,
        env=env,
        sampler_cls=sampler,
        n_workers=n_workers,
        # worker_class=DefaultWorker
    )

    trainer.eval_sampler_setup(
        sampler_cls=sampler,
        n_workers=n_workers,
        worker_class=DefaultWorker,
    )

    sac.to()

    trainer.train(n_epochs=number_epochs, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diversity is all you need')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=False)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--number_skills', type=int, required=False)
    parser.add_argument('--render_freq', type=int, default=200)
    parser.add_argument('--number_epochs', type=int, required=False)
    parser.add_argument('--snapshot_gap', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--max_episode_length', type=int, required=False)
    parser.add_argument('--alpha', type=float, required=False)
    parser.add_argument('--ray', action='store_true', required=False)
    parser.add_argument('--use_gpu', action='store_true', required=False)
    parser.add_argument('--gpu_id', type=int, default=0)

    torch.set_num_threads(4)

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}


    assert args['number_skills'] % args['n_workers'] == 0, \
        "number of skills should be set as a multiple of n_workers"

    args['name'] = (
        args['name'] + '_' + time.ctime().replace(' ', '_')
    )

    args['seed'] = (
        args['seed'] if 'seed' in args.keys() else np.random.randint(0, 1000)
    )

    logger.add_output(
        dowel.WandbOutput(
            project='diversity',
            name=args['name'],
            config=args,

        )
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           args['name'])

    diversity_is_all_you_need = wrap_experiment(
        diversity_is_all_you_need,
        name=args['name'],
        snapshot_gap=args['snapshot_gap'],
        snapshot_mode='gapped_last',
        log_dir=log_dir,
    )

    diversity_is_all_you_need(**args)
