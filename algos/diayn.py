import torch
import numpy as np
from dowel import tabular

from garage import StepType
from garage.torch.algos import SAC
from garage.torch import dict_np_to_torch, global_device
from garage import log_performance, obtain_evaluation_episodes
from garage import EpisodeBatch

import torch.nn.functional as F

import pickle


def get_skill_ints(observations, number_of_skills):
    skill_one_hot = observations[:, :number_of_skills]
    skills = np.tile(
        np.arange(number_of_skills),
        (observations.shape[0], 1)
    )[skill_one_hot.astype(bool)].flatten()
    return skills


class DIAYN(SAC):

    def __init__(
        self,
        env_spec,
        policy,
        qf1,
        qf2,
        discriminator,
        replay_buffer,
        *,  # Everything after this is numbers.
        number_skills=10,
        collect_skill_freq,
        max_episode_length_eval=None,
        gradient_steps_per_itr,
        discriminator_gradient_steps_per_itr,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=int(1e4),
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        discriminator_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        steps_per_epoch=1,
        num_evaluation_episodes=10,
        eval_env=None,
        use_deterministic_evaluation=True
    ):

        super().__init__(
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            max_episode_length_eval=max_episode_length_eval,
            gradient_steps_per_itr=gradient_steps_per_itr,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_episodes=num_evaluation_episodes,
            eval_env=eval_env,
            use_deterministic_evaluation=use_deterministic_evaluation)

        self.discriminator = discriminator
        self.number_skills = number_skills
        self.collect_skill_freq = collect_skill_freq
        self._discriminator_lr = discriminator_lr
        self._discriminator_gradient_steps = discriminator_gradient_steps_per_itr
        self._discriminator_optimizer = self._optimizer(
            self.discriminator.parameters(),
            lr=self._discriminator_lr)

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
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_path = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                trainer.step_path = self.update_diversity_rewards(
                    trainer.step_path)
                path_returns = []
                for path in trainer.step_path:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_path)
                self.episode_rewards.append(np.mean(path_returns))

                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
                    discriminator_loss = self.train_discriminator()

                # for _ in range(self._discriminator_gradient_steps):
                    # discriminator_loss = self.train_discriminator()

            if trainer.step_itr % 25 == 0:
                last_return = self._evaluate_policy(trainer)

            self._log_statistics(policy_loss,
                                 qf1_loss,
                                 qf2_loss,
                                 discriminator_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            trainer.step_itr += 1

        return np.mean(last_return)

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from the Policy/Actor.
        """
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(obs, new_actions),
                                      self._qf2(obs, new_actions))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()

        tabular.record('Policy/AverageEntropy',
                       -torch.mean(log_pi_new_actions).cpu().detach().numpy())

        skills = get_skill_ints(obs.cpu().detach().numpy(), self.number_skills)
        skills, counts = np.unique(skills, return_counts=True)
        tabular.record('Policy/NumberSkills', len(skills))
        tabular.record('Policy/SkillCountMax', counts.max())
        tabular.record('Policy/SkillCountMin', counts.min())

        return policy_objective

    def get_eval_episodes(self, num_eval_episodes):
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=num_eval_episodes,
            deterministic=self._use_deterministic_evaluation)
        eval_episodes = EpisodeBatch.from_list(
            self._eval_env.spec,
            self.update_diversity_rewards(eval_episodes.to_list())
        )
        return eval_episodes

    def _evaluate_policy(self, trainer):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        epoch = trainer.step_itr
        eval_episodes = self.get_eval_episodes(self._num_evaluation_episodes)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)

        if epoch % self.collect_skill_freq == 0:
            with open(trainer._snapshotter.snapshot_dir +
                      '/eps_{}'.format(epoch), 'wb') as f:
                pickle.dump(eval_episodes, f)

        return last_return

    def train_discriminator(self):
        self._discriminator_optimizer.zero_grad()
        samples = self.replay_buffer.sample_transitions(
            self._buffer_batch_size)
        samples = dict_np_to_torch(samples)
        observations = samples['observation']
        skills_one_hot = observations[:, :self.number_skills]
        skills = (torch.arange(self.number_skills)
                       .reshape(1, -1)
                       .repeat(observations.shape[0], 1)
                  )[skills_one_hot.type(torch.bool)]

        states = samples['observation'][:, self.number_skills:]

        log_probs = self.discriminator(states)
        loss = F.nll_loss(log_probs, skills)
        loss.backward()
        self._discriminator_optimizer.step()
        return loss

    def _log_statistics(self,
                        policy_loss,
                        qf1_loss,
                        qf2_loss,
                        discriminator_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('Discriminator/Loss', float(discriminator_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    def diversity_reward(self, observations, skills):
        log_q = self.discriminator(
            torch.tensor(observations.astype(np.float32)).to(global_device())
        )[range(observations.shape[0]), skills]

        log_p = torch.log(torch.tensor(1/self.number_skills))

        return log_q - log_p

    def update_diversity_rewards(self, step_path):
        lengths = []
        observations = []
        for path in step_path:
            observations.append(path['observations'])
            lengths.append(path['observations'].shape[0])
        observations = np.concatenate(observations, axis=0)

        states = observations[:, self.number_skills:]
        skills_one_hot = observations[:, :self.number_skills]
        skills = np.repeat(np.arange(self.number_skills).reshape(1, -1),
                           observations.shape[0],
                           axis=0)[skills_one_hot.astype(bool)]

        with torch.no_grad():
            rewards = self.diversity_reward(states, skills).numpy()

        start = 0
        for length, path in zip(lengths, step_path):
            path['rewards'] = np.array(rewards[start:start+length])
            start += length

        return step_path
