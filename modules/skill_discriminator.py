from typing import Dict

from garage.torch import dict_np_to_torch, global_device, np_to_torch
from garage.torch.modules import MLPModule

import numpy as np

import torch
import torch.nn.functional as F

from utils import split_flattened, one_hot_to_int


class SkillDiscriminator(MLPModule):
    """
    SkillDiscriminator is a Module to discriminate amongst skills in the
    diversity is all you need algorithm
    """

    def __init__(self,
                 env_spec,
                 num_skills,
                 learning_rate,
                 **kwargs):
        """TODO: to be defined.

        Parameters
        ----------
        env_spec : TODO
        num_skills : TODO
        optimizer : TODO


        """
        super().__init__(
            input_dim=env_spec.observation_space['state'].flat_dim,
            output_dim=num_skills,
            **kwargs,
        )

        self._env_spec = env_spec
        self._num_skills = num_skills
        self._optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
        )

    def train_once(
            self,
            samples: Dict[str, torch.Tensor]):
        """
        TODO
        """
        self._optimizer.zero_grad()
        del samples['reward']
        space = self._env_spec.observation_space
        observations = split_flattened(space, samples['observations'])
        skills_one_hot = observations['skill']
        states = observations['state']
        skills = one_hot_to_int(skills_one_hot).to(global_device())

        logits = self(states)
        loss = F.cross_entropy(logits, skills)
        loss.backward()
        self._optimizer.step()
        return loss

    def diversity_reward(self,
                         states: np.ndarray,
                         skills: np.ndarray):
        """
        TODO
        """
        logits = self(np_to_torch(states))
        dist = torch.distributions.Categorical(logits=logits)
        log_q = dist.log_prob(np_to_torch(skills))
        log_p = torch.log(torch.tensor(1/self._num_skills))
        return log_q - log_p

    def update_diversity_rewards_in_buffer_samples(
            self,
            samples: Dict[str, np.ndarray]):
        """
        TODO
        """
        space = self._env_spec.observation_space
        observations = split_flattened(space, samples['observations'])
        states = observations['state']
        skills_one_hot = observations['skill']
        skills = one_hot_to_int(skills_one_hot)

        with torch.no_grad():
            rewards = self.diversity_reward(states, skills).cpu().numpy()

        samples['reward'] = rewards.reshape((-1, 1))
        return samples

    def forward(self, observations):
        """Forward method.

        Args:
            observations (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(observations)
