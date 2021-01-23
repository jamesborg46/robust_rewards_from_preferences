"""Wrapper for diversity is all you need implementation for gym.Env"""
import gym
from gym import spaces
import numpy as np

from garage.torch import global_device
import torch


class DiversityWrapper(gym.Wrapper):

    def __init__(self, env,  number_skills=10, skill_mode='random'):
        super().__init__(env)

        self.number_skills = number_skills
        self.original_obs_space = env.observation_space
        self.init_new_obs_space()
        self.skill = None
        self.skill_mode = skill_mode

    def init_new_obs_space(self):
        if isinstance(self.env.observation_space, spaces.Box) and \
                len(self.env.observation_space.shape) == 1:

            obs_size = self.env.observation_space.shape[0]
            low = self.env.observation_space.low
            high = self.env.observation_space.high

            if not np.isscalar(low):
                assert high.shape == low.shape
                new_low = np.concatenate(
                    [np.zeros(self.number_skills), low]
                )
                new_high = np.concatenate(
                    [np.ones(self.number_skills), high]
                )
                new_obs_shape = None
            else:
                new_high = max(low, 1)
                new_low = min(low, 0)
                new_obs_shape = (obs_size + self.number_skills, )

            self.observation_space = spaces.Box(
                low=new_low,
                high=new_high,
                shape=new_obs_shape
            )

        else:
            raise NotImplementedError(
                'Only box spaces of one dimension handled in diversity mask')

    def set_skill_mode(self, skill_mode):
        if skill_mode not in ['random', 'consecutive', 'constant']:
            raise ValueError("skill_mode must be 'random', 'consecutive' or"
                             "constant")

        self.skill_mode = skill_mode

    def set_skill(self, skill):
        self.skill = skill

    def reset(self):
        if self.skill_mode not in ['random', 'consecutive', 'constant']:
            raise ValueError("skill_mode must be 'random', 'consecutive' or"
                             "constant")

        if self.skill_mode == 'random':
            self.skill = np.random.randint(low=0, high=self.number_skills)
        elif self.skill_mode == 'consecutive':
            self.skill = (self.skill + 1) % self.number_skills
        elif self.skill_mode == 'constant':
            pass
        else:
            raise Exception()

        self.skill_one_hot = np.zeros(self.number_skills)
        self.skill_one_hot[self.skill] = 1

        obs = self.env.reset()
        new_obs = np.concatenate([self.skill_one_hot, obs])
        return new_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['gt_reward'] = reward
        new_obs = np.concatenate([self.skill_one_hot, obs])
        return new_obs, None, done, info

