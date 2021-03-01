from dowel import tabular

import numpy as np

import torch

from garage import log_performance
from garage.torch.algos import SAC
from garage.torch import dict_np_to_torch

from modules import SkillDiscriminator
from utils import log_episodes, profile


class DIAYN(SAC):
    """Docstring for DIAYN."""

    def __init__(self,
                 number_skills: int,
                 discriminator: SkillDiscriminator,
                 snapshot_dir,
                 log_sampler,
                 render_freq=200,
                 **kwargs
                 ):
        """
        TODO: to be defined.

        Parameters
        ----------
        number_skills : TODO
        discriminator : TODO
        snapshot_dir : TODO
        log_sampler : TODO
        render_freq : TODO, optional
        """
        super().__init__(**kwargs)

        self._number_skills = number_skills
        self._discriminator = discriminator
        self._snapshot_dir = snapshot_dir
        self._log_sampler = log_sampler
        self._render_freq = render_freq

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()

        batch_size = int(self._min_buffer_size)
        eps = trainer.obtain_episodes(trainer.step_itr, batch_size)
        self.replay_buffer.add_episode_batch(eps)

        last_return = None
        for epoch in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                eps = profile(
                    'ObtainEpisodes',
                    trainer.obtain_episodes,
                    trainer.step_itr)

                self.replay_buffer.add_episode_batch(eps)
                path_returns = [sum(ep.rewards) for ep in eps.split()]
                assert len(path_returns) == len(eps.lengths)
                self.episode_rewards.append(np.mean(path_returns))

                policy_loss, qf1_loss, qf2_loss, disc_loss = profile(
                    'TrainOnce',
                    self.train_once,
                    trainer.step_itr
                )

            # last_return = profile(
            #     'EvaluatePolicy',
            #     self._evaluate_policy,
            #     trainer.step_itr)

            self._log_statistics(policy_loss, qf1_loss, qf2_loss, disc_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)

            trainer.step_itr += 1

            if trainer.step_itr % self._render_freq == 0:
                eval_episodes = log_episodes(
                    trainer.step_itr,
                    self._snapshot_dir,
                    self._log_sampler,
                    self.policy,
                    enable_render=True,
                    capture_state=True,
                    number_skills=self._number_skills)
            else:
                eval_episodes = log_episodes(
                    trainer.step_itr,
                    self._snapshot_dir,
                    self._log_sampler,
                    self.policy,
                    enable_render=False,
                    capture_state=False,
                    number_skills=self._number_skills)

            last_return = log_performance(
                epoch,
                eval_episodes,
                discount=self._discount)

            self._log_statistics(policy_loss, qf1_loss, qf2_loss, disc_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)

        return np.mean(last_return)

    def train_once(self, itr, paths=None):
        """Complectes 1 training iteration of DIAYN

        Parameters
        ----------
        itr : TODO, optional
        paths : TODO, optional

        Returns
        -------
        TODO

        """
        del paths

        policy_losses = []
        qf1_losses = []
        qf2_losses = []
        disc_losses = []

        for i in range(self._gradient_steps):
            if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
                samples = self.replay_buffer.sample_transitions(
                    self._buffer_batch_size)

                samples = (
                    self._discriminator
                    .update_diversity_rewards_in_buffer_samples(samples)
                )

                samples = dict_np_to_torch(samples)
                policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
                self._update_targets()
                disc_loss = self._discriminator.train_once(samples)

                policy_losses.append(policy_loss)
                qf1_losses.append(qf1_loss)
                qf2_losses.append(qf2_loss)
                disc_losses.append(disc_loss)

        policy_loss = torch.stack(policy_losses).mean().item()
        qf1_loss = torch.stack(qf1_losses).mean().item()
        qf2_loss = torch.stack(qf2_losses).mean().item()
        disc_loss = torch.stack(disc_losses).mean().item()

        return policy_loss, qf1_loss, qf2_loss, disc_loss

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss, disc_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Policy/Loss', policy_loss)
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('DiscriminatorLoss', float(disc_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2, self._discriminator
        ]

