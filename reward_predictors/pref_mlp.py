"""MLP Reward Predictor"""
import torch
from torch import nn
from dowel import logger

from garage import EpisodeBatch, TimeStepBatch
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
                 iterations_per_epoch=100,
                 pretrain_epochs=10,
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
            obs = np_to_torch(obs_space.flatten_n(steps.observations))
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
