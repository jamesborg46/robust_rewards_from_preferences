"""MLP Reward Predictor"""
import torch
from torch import nn
from dowel import logger
import numpy as np

from garage.torch import np_to_torch
from garage.torch.modules import MLPModule
from garage.torch.optimizers import OptimizerWrapper
from reward_predictors import RewardPredictor


class PrefMLP(nn.Module, RewardPredictor):
    """Gaussian MLP Value Function with Model.
    It fits the input data to a gaussian distribution estimated by
    a MLP.
    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.
    """

    def __init__(self,
                 env_spec,
                 preference_collector,
                 learning_rate=0.001,
                 minibatch_size=256,
                 max_optimization_epochs=100,
                 pretrain_epochs=1000,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='PrefMLPRewardPredictor'):

        super().__init__()

        self.env_spec = env_spec
        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.name = name

        self.module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self.pretrain_epochs = pretrain_epochs
        self.preference_collector = preference_collector
        self.optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=learning_rate)),
            self.module,
            max_optimization_epochs=max_optimization_epochs,
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

    def pretrain(self, trainer):
        logger.log('Pretraining reward predictor...')

        # Acquiring trajectories
        logger.log('Acquiring pre-train episodes')
        while not self.preference_collector.buffer_full:
            paths = trainer.obtain_samples(itr=0)
            self.preference_collector.collect(paths)

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

    def train_once(self, itr, paths):
        self._train_once()
        self.preference_collector.collect(paths)

    def _scale_rewards(self, rewards):
        scaled_rewards = (
            -3 + 2*(rewards - rewards.min()) / (rewards.max() - rewards.min())
        )
        return scaled_rewards

    def predict_rewards(self, itr, paths):
        obs_space = self.env_spec.observation_space
        lengths = np.array([len(path['observations']) for path in paths])

        with torch.no_grad():
            obs = np_to_torch(np.concatenate(
                [obs_space.flatten_n(path['observations']) for path in paths]))

            predicted_rewards = self.forward(obs).numpy()

        predicted_rewards = self._scale_rewards(predicted_rewards)
        predicted_rewards = np.split(predicted_rewards, np.cumsum(lengths)[:-1])
        assert len(predicted_rewards) == len(paths)

        for path, predicted_reward in zip(paths, predicted_rewards):
            assert len(path['rewards']) == len(predicted_reward)
            path['rewards'] = predicted_reward.flatten()

        return paths

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.
        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.
        """
        return self.module(obs)
