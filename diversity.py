import argparse
import json
import os
import pickle
import time

import dowel
from dowel import logger
import gym
import numpy as np
import safety_gym  # noqa: F401
from safety_gym.envs.engine import Engine
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer  # noqa: F401
from garage.sampler import LocalSampler, RaySampler, \
    DefaultWorker  # noqa: F401
from garage.torch import set_gpu_mode
from garage.torch.policies import TanhGaussianMLPPolicy  # noqa: F401
from garage.torch.q_functions import ContinuousMLPQFunction  # noqa: F401
from garage.torch.modules import MLPModule  # noqa: F401
from garage.trainer import Trainer

from algos import DIAYN  # noqa: F401
import envs.custom_safety_envs  # noqa: F401
from wrappers import DiversityWrapper, SafetyEnvStateAppender, Renderer


def diversity_is_all_you_need(ctxt=None,
                              **kwargs,
                              ):

    kwargs['number_epochs'] += 1

    snapshot_dir = ctxt.snapshot_dir
    env = gym.make(kwargs['env_id'])

    if isinstance(env, Engine):
        config = env.config

        with open(os.path.join(snapshot_dir, 'config.json'), 'w') \
                as outfile:
            json.dump(config, outfile)

    env.metadata['render.modes'] = ['rgb_array']
    env = DiversityWrapper(env, number_skills=kwargs['number_skills'])
    env = SafetyEnvStateAppender(env)
    env = Renderer(env, directory=os.path.join(snapshot_dir, 'videos'))
    env = GymEnv(env, max_episode_length=kwargs['max_episode_length'])
    set_seed(kwargs['seed'])

    with open(os.path.join(snapshot_dir, 'env.pkl'), 'wb') as outfile:
        pickle.dump(env, outfile)

    trainer = Trainer(ctxt)
    skill_discriminator = eval(kwargs['skill_discriminator'])  # noqa: F841
    discrminator_optimizer = eval(  # noqa: F841
        kwargs['discrminator_optimizer'])
    policy = eval(kwargs['policy'])  # noqa: F841
    qf1 = eval(kwargs['qf'])  # noqa: F841
    qf2 = eval(kwargs['qf'])  # noqa: F841
    replay_buffer = eval(kwargs['replay_buffer'])  # noqa: F841

    if torch.cuda.is_available() and kwargs['use_gpu']:
        set_gpu_mode(True, gpu_id=kwargs['gpu_id'])
    else:
        set_gpu_mode(False)

    Sampler = RaySampler if kwargs['ray'] else LocalSampler
    sampler = Sampler(agents=policy,  # noqa: F841
                      envs=env,
                      max_episode_length=kwargs['max_episode_length'],
                      n_workers=kwargs['n_workers'])
    log_sampler = Sampler(agents=policy,  # noqa: F841
                          envs=env,
                          max_episode_length=kwargs['max_episode_length'],
                          n_workers=kwargs['n_workers'])

    diayn = eval(kwargs['diayn'])

    trainer.setup(
        algo=diayn,
        env=env,
    )

    diayn.to()

    trainer.train(n_epochs=kwargs['number_epochs'],
                  batch_size=kwargs['steps_per_epoch'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diversity is all you need')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=False)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--number_skills', type=int, required=False)
    parser.add_argument('--render_freq', type=int, default=200)
    parser.add_argument('--number_epochs', type=int, required=False)
    parser.add_argument('--snapshot_gap', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--max_episode_length', type=int, required=False)
    parser.add_argument('--alpha', type=float, required=False)
    parser.add_argument('--ray', action='store_true', required=False)
    parser.add_argument('--use_gpu', action='store_true', required=False)
    parser.add_argument('--gpu_id', type=int, default=0)

    torch.set_num_threads(4)

    kwargs = vars(parser.parse_args())
    args = {k: v for k, v in kwargs.items() if v is not None}

    assert kwargs['number_skills'] % kwargs['n_workers'] == 0, \
        "number of skills should be set as a multiple of n_workers"

    kwargs['name'] = (
        kwargs['name'] + '_' + time.ctime().replace(' ', '_')
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           kwargs['name'] + time.ctime().replace(' ', '_'))

    logger.add_output(
        dowel.WandbOutput(
            project='diversity',
            name=args['name'],
            config=kwargs,
        )
    )

    diversity_is_all_you_need = wrap_experiment(
        diversity_is_all_you_need,
        name=args['name'],
        snapshot_gap=args['snapshot_gap'],
        snapshot_mode='gap_overwrite',
        log_dir=log_dir,
    )

    diversity_is_all_you_need(**args)
