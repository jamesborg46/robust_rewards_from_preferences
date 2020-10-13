#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
# from garage.torch.modules import MLPModule
from garage.sampler import LocalSampler
from garage.trainer import Trainer

import gym
import safety_gym
from gym.wrappers import Monitor
from wrappers import RewardMasker
from algos import PreferenceTRPO
from buffers import ComparisonBuffer, LabelAnnealer
from reward_predictors import MLPRewardPredictor
# gym.logger.set_level(10)


@wrap_experiment
def trpo_pendulum(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    n_epochs = 2005

    set_seed(seed)
    # naked_env = gym.make('InvertedDoublePendulum-v2')
    naked_env = gym.make('Safexp-PointGoal1-v0')
    naked_env.metadata['render.modes'] = ['rgb_array']
    naked_env = Monitor(naked_env, 'monitoring')

    env = GymEnv(RewardMasker(naked_env),
                 max_episode_length=1000)

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
        hidden_sizes=(32, 32),
    )

    label_scheduler = LabelAnnealer(final_timesteps=n_epochs,
                                    final_labels=800,
                                    pretrain_labels=200)

    comparison_buffer = ComparisonBuffer(20000, label_scheduler)

    algo = PreferenceTRPO(env_spec=env.spec,
                          policy=policy,
                          value_function=value_function,
                          reward_predictor=reward_predictor,
                          comparison_buffer=comparison_buffer,
                          discount=0.99,
                          center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=1024)


trpo_pendulum(seed=1)