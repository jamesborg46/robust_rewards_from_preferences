import torch
import numpy as np
from dowel import tabular
from collections import OrderedDict

from garage import StepType
from garage.torch.algos import SAC
from garage.torch import dict_np_to_torch, global_device
from garage import log_performance, obtain_evaluation_episodes
from garage import EpisodeBatch
from garage.sampler.env_update import EnvUpdate

import torch.nn.functional as F

import pickle
import os
import os.path as osp
from utils.video import export_video
from collections import defaultdict

def get_skill_ints(observations, number_of_skills):
    skill_one_hot = observations[:, :number_of_skills]
    skills = np.tile(
        np.arange(number_of_skills),
        (observations.shape[0], 1)
    )[skill_one_hot.astype(bool)].flatten()
    return skills


def agent_update(policy, device='cpu'):
    state_dict = OrderedDict(
        [(k, v.to(device)) for k, v in policy.state_dict().items()]
    )
    return state_dict


class EnvConfigUpdate(EnvUpdate):

    def __init__(self,
                 capture_state=False,
                 capture_render=False,
                 skill_mode='random',
                 skill=None):

        self.capture_state = capture_state
        self.capture_render = capture_render
        self.skill_mode = skill_mode
        self.skill = skill

    def __call__(self, old_env):
        old_env.set_capture_state(self.capture_state)
        old_env.set_capture_render(self.capture_render)
        old_env.set_skill_mode(self.skill_mode)
        old_env.set_skill(self.skill)
        return old_env


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
                    trainer.step_itr,
                    batch_size,
                    agent_update=agent_update(self.policy),
                )

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

                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
                    discriminator_loss = self.train_discriminator()

            if trainer.step_itr and trainer.step_itr % 2 == 0:
                self._log_episodes(trainer,
                                   capture_render=False,
                                   capture_state=True)

            self._log_statistics(policy_loss,
                                 qf1_loss,
                                 qf2_loss,
                                 discriminator_loss)

            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            tabular.record(
                'Policy/AverageActionStd',
                np.mean([np.exp(path['agent_infos']['log_std']) for
                         path in trainer.step_path])
            )

            trainer.step_itr += 1

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)
            samples = self.update_diversity_rewards_in_buffer_samples(samples)
            samples = dict_np_to_torch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

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

    def get_eval_episodes(self,
                          sampler,
                          n_workers,
                          episodes_per_skill=1,
                          capture_render=False,
                          capture_state=False):

        assert self.number_skills % n_workers == 0
        skills_per_worker = self.number_skills / n_workers

        env_updates = []
        for i in range(n_workers):
            env_updates.append(EnvConfigUpdate(
                capture_state=capture_state,
                capture_render=capture_render,
                skill_mode='consecutive',
                skill=int(i*skills_per_worker),
            ))

        sampler._update_workers(
            agent_update=agent_update(self.policy),
            env_update=env_updates
        )

        episodes = sampler.obtain_exact_episodes(
            n_eps_per_worker=skills_per_worker*episodes_per_skill,
            agent_update=agent_update(self.policy)
        )

        return episodes

    def _log_episodes(self,
                      trainer,
                      capture_render=False,
                      capture_state=False):
        """
        Retrieves evaluation episodes and can render into videos or log as pkl
        files
        """
        eval_episodes = self.get_eval_episodes(
            sampler=trainer._eval_sampler,
            n_workers=trainer._eval_n_workers,
            episodes_per_skill=2,
            capture_render=capture_render,
            capture_state=capture_state)

        if capture_render:
            self._render_skills(
                eval_episodes,
                osp.join(trainer._snapshotter.snapshot_dir, 'videos'),
                trainer.step_itr
            )

        if capture_state:
            fname = osp.join(trainer._snapshotter.snapshot_dir,
                             'episode_logs',
                             f'episode_{trainer.step_itr}.pkl')

            if not osp.isdir(osp.dirname(fname)):
                os.makedirs(osp.dirname(fname))

            with open(fname, 'wb') as f:
                pickle.dump(eval_episodes, f)

    def _render_skills(self, episodes, directory, epoch):
        episodes_per_skill = defaultdict(int)
        for ep in episodes.to_list():
            frames = (
                [f[::-1, :] for f in ep["env_infos"]['render']]
            )

            skill = int(get_skill_ints(
                ep['observations'],
                self.number_skills
            )[0])

            fname = osp.join(
                directory,
                f'epoch_{epoch}',
                f'skill_{skill:03}_ep_{episodes_per_skill[skill]:02}.mp4'
            )

            episodes_per_skill[skill] += 1

            if not osp.isdir(osp.dirname(fname)):
                os.makedirs(osp.dirname(fname))

            export_video(frames, fname)

    def train_discriminator(self):
        self._discriminator_optimizer.zero_grad()
        samples = self.replay_buffer.sample_transitions(
            self._buffer_batch_size)
        del samples['reward']
        samples = dict_np_to_torch(samples)
        observations = samples['observation']
        skills_one_hot = observations[:, :self.number_skills]
        skills = (torch.arange(self.number_skills)
                       .reshape(1, -1)
                       .repeat(observations.shape[0], 1)
                  )[skills_one_hot.type(torch.bool)].to(global_device())

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
        # tabular.record('Average/TrainAverageReturn',
        #                np.mean(self.episode_rewards))

    def diversity_reward(self, observations, skills):
        log_q = self.discriminator(
            torch.tensor(observations.astype(np.float32)).to(global_device())
        )[range(observations.shape[0]), skills]

        log_p = torch.log(torch.tensor(1/self.number_skills))

        return log_q - log_p

    def update_diversity_rewards_in_buffer_samples(self, path):
        observations = path['observation']
        states = observations[:, self.number_skills:]
        skills_one_hot = observations[:, :self.number_skills]
        skills = np.repeat(np.arange(self.number_skills).reshape(1, -1),
                           observations.shape[0],
                           axis=0)[skills_one_hot.astype(bool)]

        with torch.no_grad():
            rewards = self.diversity_reward(states, skills).cpu().numpy()

        path['reward'] = rewards.reshape((-1, 1))
        return path

    def update_diversity_rewards_in_step_path(self, step_path):
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
            rewards = self.diversity_reward(states, skills).cpu().numpy()

        start = 0
        for length, path in zip(lengths, step_path):
            path['rewards'] = np.array(rewards[start:start+length])
            start += length

        return step_path

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2, self.discriminator
        ]

