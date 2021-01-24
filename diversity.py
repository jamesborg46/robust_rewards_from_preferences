import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import gym
import safety_gym
from safety_gym.envs.engine import Engine
from gym.wrappers import Monitor
from gym.envs.registration import register
import envs.custom_safety_envs

from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler, RaySampler, DefaultWorker
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from garage.torch.modules import MLPModule

from wrappers import DiversityWrapper, SafetyEnvStateAppender
from algos import DIAYN

import json
from datetime import datetime
import time
import argparse
import os

import pickle


def diversity_is_all_you_need(ctxt=None,
                              seed=1,
                              name='EXP',
                              number_skills=10,
                              number_epochs=1000,
                              max_episode_length=1000,
                              alpha=0.1,
                              n_workers=8,
                              env_id='Safexp-PointGoal0-v0',
                              batch_size=4000,
                              ray=False,
                              use_gpu=False,
                              **kwargs,
                              ):

    set_seed(seed)
    env = gym.make(env_id)

    if isinstance(env, Engine):
        config = env.config

        with open(os.path.join(ctxt.snapshot_dir, 'config.json'), 'w') as outfile:
            json.dump(config, outfile)

    env.metadata['render.modes'] = ['rgb_array']

    env = DiversityWrapper(env, number_skills=number_skills)
    env = SafetyEnvStateAppender(env)
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
                collect_skill_freq=250,
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
        set_gpu_mode(True)
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
        sampler_cls = sampler,
        n_workers=n_workers,
        worker_class=DefaultWorker,
    )

    sac.to()

    trainer.train(n_epochs=number_epochs, batch_size=batch_size)


# def register_envs():

#     base_config = gym.make('Safexp-PointGoal1-v0').config
#     new_config = {
#         'robot_locations': [(0, -1.5)],
#         'robot_rot': np.pi * (1/2),
#         'goal_locations': [(0, 1.5)],
#         'hazards_num': 3,
#         'vases_num': 0,
#         'observe_vases': False,
#         'hazards_placements': None,
#         'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 0)],
#         'hazards_size': 0.4,
#     }

#     for k, v in new_config.items():
#         assert k in base_config.keys(), 'BAD CONFIG'
#         base_config[k] = v

#     register(id='Safexp-PointGoalCustom0-v0',
#              entry_point='safety_gym.envs.mujoco:Engine',
#              kwargs={'config': base_config})

#     base_config = gym.make('Safexp-PointGoal1-v0').config
#     new_config = {
#             'robot_locations': [(0, -1.5)],
#             'robot_rot': np.pi * (1/2),
#             'goal_locations': [(0, 1.5)],
#             'hazards_num': 1,
#             'vases_num': 1,
#             'pillars_num': 1,
#             'observe_vases': True,
#             'observe_pillars': True,
#             'constrain_vases': True,
#             'constrain_pillars': True,
#             'hazards_placements': None,
#             'hazards_locations': [(0, 0)],
#             'vases_placements': None,
#             'vases_locations': [(-1.3, 0.9)],
#             'pillars_placements': None,
#             'pillars_locations': [(1.3, 0.9)],
#             'hazards_size': 0.4,
#             'pillars_size': 0.2,
#             'vases_size': 0.2,
#             'hazards_cost': 30,
#             'vases_contact_cost': 30,
#             'pillars_cost': 30,
#     }

#     for k, v in new_config.items():
#         assert k in base_config.keys(), 'BAD CONFIG'
#         base_config[k] = v

#     register(id='Safexp-PointGoalThree0-v0',
#              entry_point='safety_gym.envs.mujoco:Engine',
#              kwargs={'config': base_config})

#     base_config = gym.make('Safexp-PointGoal1-v0').config
#     new_config = {
#         'robot_locations': [(0, -1.5)],
#         'robot_rot': np.pi * (1/2),
#         'goal_locations': [(0, 0)],
#         'hazards_num': 3,
#         'vases_num': 0,
#         'observe_vases': False,
#         'hazards_placements': None,
#         'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 1.5)],
#         'hazards_size': 0.4,
#         'hazards_cost': 30,
#     }

#     for k, v in new_config.items():
#         assert k in base_config.keys(), 'BAD CONFIG'
#         base_config[k] = v

#     register(id='Safexp-PointGoalBehind0-v0',
#              entry_point='safety_gym.envs.mujoco:Engine',
#              kwargs={'config': base_config})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diversity is all you need')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=False)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--number_skills', type=int, required=False)
    parser.add_argument('--number_epochs', type=int, required=False)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--max_episode_length', type=int, required=False)
    parser.add_argument('--alpha', type=float, required=False)
    parser.add_argument('--ray', action='store_true', required=False)
    parser.add_argument('--use_gpu', action='store_true', required=False)

    torch.set_num_threads(4)

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    assert args['number_skills'] % args['n_workers'] == 0, \
        "number of skills should be set as a multiple of n_workers"

#     config = {
#         'continue_goal': False,
#         'reward_includes_cost': True,
#         'hazards_cost': 30.,
#         'constrain_indicator': False,
#         'reward_clip': 20,
#         'reward_goal': 0,
#     }

    args['name'] = (
        args['name'] + '_' + time.ctime().replace(' ', '_')
    )

    # kwargs = {**args, **config}
    kwargs = {**args}

    # register_envs()

    args['seed'] = (
        args['seed'] if 'seed' in args.keys() else np.random.randint(0, 1000)
    )

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           args['name'])

    diversity_is_all_you_need = wrap_experiment(
        diversity_is_all_you_need,
        name=args['name'],
        snapshot_gap=50,
        snapshot_mode='gapped_last',
        log_dir=log_dir,
    )

    diversity_is_all_you_need(**kwargs)
