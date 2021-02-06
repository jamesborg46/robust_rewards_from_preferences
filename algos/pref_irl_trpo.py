"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO
from garage.torch import pad_to_last, filter_valids
from garage.np import discount_cumsum
from garage import StepType, EpisodeBatch
from garage.torch.optimizers import OptimizerWrapper
from dowel import tabular, logger

import torch
import numpy as np
import time
import pickle


class PreferenceTRPO(TRPO):

    def __init__(self,
                 reward_predictor,
                 reward_predictor_frozen=False,
                 # comparison_collector,
                 # test_comparison_collector,
                 val_freq=100,
                 reward_predictor_optimizer=None,
                 val_opt_its=100,
                 val_opt_lr=1e-3,
                 reward_opt_its=20,
                 reward_opt_lr=1e-3,
                 **kwargs,
                 ):

        super().__init__(**kwargs)

        self.forward_algo = forward_algo
        self.policy = forward_algo.policy

        self._reward_predictor = reward_predictor
        self._reward_predictor_frozen = reward_predictor_frozen
        self._comparison_collector = comparison_collector
        self._test_comparison_collector = test_comparison_collector
        self._val_freq = val_freq
        self._totally_ordered = totally_ordered

        if reward_predictor_optimizer:
            self._reward_predictor_optimizer = reward_predictor_optimizer
        else:
            self._reward_predictor_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=reward_opt_lr)),
                self._reward_predictor,
                max_optimization_epochs=reward_opt_its
            )
        self.reward_predictor_its = reward_opt_its

    def train(self, trainer):

        self._pretrain_reward(trainer)

        super().train(trainer)

    def _train(self,
               left_segs,
               right_segs,
               **kwargs):

        super()._train()

    def _train_once(self, itr, paths):

        obs, actions, rewards, returns, valids, baselines = \
            self._process_samples(paths)

        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        with torch.no_grad():
            pass

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        with torch.no_grad():
            pass

        return super()._train_once(itr, paths)


