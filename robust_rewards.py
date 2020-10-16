#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
# from garage.torch.modules import MLPModule
from garage.sampler import LocalSampler
from garage.trainer import Trainer

import gym
import safety_gym
from gym.wrappers import Monitor
from wrappers import RewardMasker, SafetyEnvStateAppender
from algos import PreferenceTRPO
from buffers import SyntheticComparisonCollector, HumanComparisonCollector, LabelAnnealer
from reward_predictors import MLPRewardPredictor
# gym.logger.set_level(10)

from datetime import datetime
import argparse


@wrap_experiment
def robust_preferences(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    parser = argparse.ArgumentParser(description='preference reward learning')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, default='Safexp-PointGoal1-v0')
    args = parser.parse_args()

    n_epochs = 2005

    set_seed(seed)
    env_id = args.env_id
    experiment_name = (
        args.name + '_' +
        env_id + '_' +
        datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
    )

    # env_id = 'InvertedDoublePendulum-v2'
    env = gym.make(env_id)
    env.metadata['render.modes'] = ['rgb_array']
    # env = Monitor(env, 'monitoring')
    env = RewardMasker(env)
    env = SafetyEnvStateAppender(env)
    env = GymEnv(env, max_episode_length=1000)

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
                                    final_labels=1600,
                                    pretrain_labels=200)

    # comparison_collector = SyntheticComparisonCollector(20000, label_scheduler)
    comparison_collector = HumanComparisonCollector(
        20000,
        label_scheduler,
        env=env,
        experiment_name=experiment_name)

    algo = PreferenceTRPO(env_spec=env.spec,
                          policy=policy,
                          value_function=value_function,
                          reward_predictor=reward_predictor,
                          comparison_collector=comparison_collector,
                          discount=0.99,
                          center_adv=False)

    trainer.setup(algo,
                  env,
                  sampler_cls=LocalSampler,
                  )
    trainer.train(n_epochs=n_epochs, batch_size=1024)


robust_preferences(seed=1)
