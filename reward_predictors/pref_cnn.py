"""MLP Reward Predictor"""
import torch
from torch import nn
from dowel import logger

from garage import EpisodeBatch, TimeStepBatch
from garage.torch import np_to_torch
from garage.torch.modules import DiscreteCNNModule
from garage.torch.optimizers import OptimizerWrapper
from reward_predictors import RewardPredictor


class PrefCNN(DiscreteCNNModule, RewardPredictor):

    def __init__(self,
                 env_spec,
                 preference_collector,
                 kernel_sizes,
                 hidden_channels,
                 strides,
                 learning_rate=0.001,
                 minibatch_size=256,
                 iterations_per_epoch=100,
                 pretrain_epochs=10,
                 hidden_sizes=(32, 32),
                 cnn_hidden_nonlinearity=torch.nn.ReLU,
                 mlp_hidden_nonlinearity=torch.nn.ReLU,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 pool_stride=1,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 is_image=True,
                 name='PrefCNNRewardPredictor'):

        self.env_spec = env_spec
        input_shape = (1, ) + env_spec.observation_space.shape
        output_dim = 1

        self.name = name

        super().__init__(input_shape=input_shape,
                         output_dim=output_dim,
                         kernel_sizes=kernel_sizes,
                         strides=strides,
                         hidden_sizes=hidden_sizes,
                         hidden_channels=hidden_channels,
                         cnn_hidden_nonlinearity=cnn_hidden_nonlinearity,
                         mlp_hidden_nonlinearity=mlp_hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         paddings=paddings,
                         padding_mode=padding_mode,
                         max_pool=max_pool,
                         pool_shape=pool_shape,
                         pool_stride=pool_stride,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         layer_normalization=layer_normalization,
                         is_image=is_image)

        self.pretrain_epochs = pretrain_epochs
        self.preference_collector = preference_collector
        self.optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=learning_rate)),
            self.module,
            max_optimization_epochs=iterations_per_epoch,
            minibatch_size=minibatch_size,
        )

    def predict_preferences(self, left, right):
        if not left.shape == right.shape:
            raise ValueError('Left and Right should have same shape')
        assert left.ndim == 3
        batch, timesteps, obs_dim = left.shape
        assert timesteps == 1  # 1 Time step per segment

        left = left.reshape(batch, obs_dim)
        right = right.reshape(batch, obs_dim)

        left_out = self.module(left)
        right_out = self.module(right)
        logits = torch.cat([left_out, right_out], dim=1)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def pretrain(self, trainer, precollected_eps=None):
        logger.log('Pretraining reward predictor...')

        # Acquiring trajectories
        logger.log('Acquiring pre-train episodes')
        if precollected_eps is not None:
            self.preference_collector.collect(precollected_eps)

        while not self.preference_collector.buffer_full:
            eps = trainer.obtain_episodes(itr=0)
            self.preference_collector.collect(eps)

        # Acquiring comparisons
        logger.log('Acquiring pretrain comparisons')
        self.preference_collector.sample_comparisons(itr=0)

        # Acquiring labels
        logger.log('Labelling pretrain comparisons')
        self.preference_collector.label_unlabeled_comparisons()

        for _ in range(self.pretrain_epochs):
            self._train_once()

    def _train_once(self):
        left, right, prefs = (
            self.preference_collector.get_labelled_preferences()
        )

        left, right, prefs = [np_to_torch(arr) for arr in [left, right, prefs]]
        prefs = prefs.type(torch.long)

        for left_batch, right_batch, prefs_batch \
                in self.optimizer.get_minibatch(left, right, prefs):
            self.optimizer.zero_grad()
            logits, preds = self.predict_preferences(left_batch,
                                                     right_batch)
            loss = nn.functional.cross_entropy(logits, prefs_batch)
            loss.backward()
            self.optimizer.step()

    def train_once(self, itr, eps):
        self._train_once()
        self.preference_collector.collect(eps)

    def _scale_rewards(self, rewards):
        scaled_rewards = (
            -3 + 2*(rewards - rewards.min()) / (rewards.max() - rewards.min())
        )
        return scaled_rewards

    def predict_rewards(self, itr, steps):
        obs_space = self.env_spec.observation_space

        with torch.no_grad():
            predicted_rewards = self.forward(obs).numpy().flatten()

        predicted_rewards = self._scale_rewards(predicted_rewards)
        assert len(predicted_rewards) == len(steps.observations)

        if type(steps) == EpisodeBatch:
            return EpisodeBatch(env_spec=steps.env_spec,
                                episode_infos=steps.episode_infos,
                                observations=steps.observations,
                                last_observations=steps.last_observations,
                                actions=steps.actions,
                                rewards=predicted_rewards,
                                env_infos=steps.env_infos,
                                agent_infos=steps.agent_infos,
                                step_types=steps.step_types,
                                lengths=steps.lengths)
        elif type(steps) == TimeStepBatch:
            return TimeStepBatch(env_spec=steps.env_spec,
                                 episode_infos=steps.episode_infos,
                                 observations=steps.observations,
                                 next_observations=steps.next_observations,
                                 actions=steps.actions,
                                 rewards=predicted_rewards,
                                 env_infos=steps.env_infos,
                                 agent_infos=steps.agent_infos,
                                 step_types=steps.step_types)

    # pylint: disable=arguments-differ
    def forward(self, observations):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations of shape :math: `(N, O*)`.

        Returns:
            torch.Tensor: Output value
        """
        if observations.shape != self._env_spec.observation_space.shape:
            # avoid using observation_space.unflatten_n
            # to support tensors on GPUs
            obs_shape = ((len(observations), ) +
                         self._env_spec.observation_space.shape)
            observations = observations.reshape(obs_shape)
        return super().forward(observations)
