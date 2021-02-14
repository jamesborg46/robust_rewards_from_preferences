"""This modules creates a DDPG model in PyTorch."""
import collections
import copy
import time
import warnings

from utils import log_episodes, update_remote_agent_device, log_gt_performance

from dowel import logger, tabular
import numpy as np

from garage.torch import global_device
from garage.torch.algos import DQN


class IrlDQN(DQN):

    def __init__(
            self,
            reward_predictor,
            snapshot_dir,
            eval_sampler,
            render_freq,
            **kwargs):
        super().__init__(**kwargs)
        self._reward_predictor = reward_predictor
        self._snapshot_dir = snapshot_dir
        self._eval_sampler = eval_sampler
        self._render_freq = render_freq

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_returns = [float('nan')]

        if self._min_buffer_size > self.replay_buffer.n_transitions_stored:
            num_warmup_steps = (self._min_buffer_size -
                                self.replay_buffer.n_transitions_stored)
            eps = trainer.obtain_episodes(
                0,
                num_warmup_steps,
                agent_update=update_remote_agent_device(
                    self.exploration_policy, device='cpu')
            )
            self.replay_buffer.add_episode_batch(eps)

        self._reward_predictor.pretrain(trainer, eps)

        trainer.enable_logging = True

        for epoch in trainer.step_epochs():
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                logger.log('Evaluating policy')
                _eval_start_time = time.time()

                params_before = self.exploration_policy.get_param_values()
                eval_eps = log_episodes(
                    itr=epoch,
                    snapshot_dir=self._snapshot_dir,
                    sampler=self._eval_sampler,
                    policy=(self.exploration_policy
                            if not self._deterministic_eval
                            else self.policy),
                    number_eps=self._num_eval_episodes,
                    capture_state=False,
                    enable_render=False)

                self.exploration_policy.set_param_values(params_before)

                last_returns = log_gt_performance(trainer.step_itr,
                                                  eval_eps,
                                                  discount=self._discount)
                self._episode_reward_mean.extend(last_returns)
                tabular.record('Evaluation/Time',
                               time.time() - _eval_start_time)
                tabular.record('Evaluation/100EpRewardMean',
                               np.mean(self._episode_reward_mean))

                self._reward_predictor.train_once(trainer.step_itr,
                                                  eval_eps)

            self._times = collections.defaultdict(list)
            for _ in range(self._steps_per_epoch):
                _step_start_time = time.time()
                trainer.step_path = trainer.obtain_episodes(
                    trainer.step_itr,
                    agent_update=update_remote_agent_device(
                        self.exploration_policy, device='cpu')
                )
                self._times['obtain_episodes'].append(
                    time.time() - _step_start_time)
                if hasattr(self.exploration_policy, 'update'):
                    self.exploration_policy.update(trainer.step_path)

                _train_once_start = time.time()
                self._train_once(trainer.step_itr, trainer.step_path)
                self._times['train_once'].append(
                    time.time() - _train_once_start)
                self._times['iter'].append(time.time() - _step_start_time)
                trainer.step_itr += 1

            if epoch and epoch % self._render_freq == 0:
                params_before = self.exploration_policy.get_param_values()
                eval_eps = log_episodes(
                    itr=trainer.step_itr,
                    snapshot_dir=self._snapshot_dir,
                    sampler=self._eval_sampler,
                    policy=(self.exploration_policy
                            if not self._deterministic_eval
                            else self.policy),
                    capture_state=False,
                    enable_render=True)
                self.exploration_policy.set_param_values(params_before)

            tabular.record('Time/TotalObtainEpisodes',
                           sum(self._times['obtain_episodes']))
            tabular.record('Time/AvgObtainEpisodes',
                           np.mean(self._times['obtain_episodes']))
            tabular.record('Time/TotalTrainOnce',
                           sum(self._times['train_once']))
            tabular.record('Time/AvglTrainOnce',
                           np.mean(self._times['train_once']))
            tabular.record('Time/TotalIter',
                           sum(self._times['iter']))
            tabular.record('Time/AvgIter',
                           np.mean(self._times['iter']))

            tabular.record('Time/TotalStep',
                           sum(self._times['step']))
            tabular.record('Time/AvgStep',
                           np.mean(self._times['step']))
            tabular.record('Time/TotalCopy',
                           sum(self._times['copy']))
            tabular.record('Time/AvgCopy',
                           np.mean(self._times['copy']))

        return np.mean(last_returns)

    def _train_once(self, itr, episodes):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self.replay_buffer.add_episode_batch(episodes)

        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                _step_start_time = time.time()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        'Observation.+is outside observation_space.+')
                    timesteps = self.replay_buffer.sample_timesteps(
                        self._buffer_batch_size)
                timesteps = self._reward_predictor.predict_rewards(timesteps)
                qf_loss, y, q = tuple(v.cpu().numpy()
                                      for v in self._optimize_qf(timesteps))
                self._times['step'].append(time.time() - _step_start_time)

                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

        if itr and itr % self._steps_per_epoch == 0:
            self._log_eval_results(epoch)

        _copy_time_start = time.time()
        if itr and itr % self._target_update_freq == 0:
            self._target_qf = copy.deepcopy(self._qf)
        self._times['copy'].append(time.time() - _copy_time_start)

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        super.to()
        if device is None:
            device = global_device()
        self._reward_predictor = self._reward_predictor.to(device)
