"""MLP Reward Predictor"""
import torch
from torch import nn

from modules.bnn import BayesianNN


class BNNRewardPredictor(nn.Module):
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
                 weight_prior_scale=[0.1, 0.00001],
                 bias_prior_scale=[1, 0.0001],
                 hidden_sizes=[32, 32],
                 prior_mix=1.,
                 weight_mu_mean_init=0.1,
                 weight_mu_scale_init=0.1,
                 weight_rho_mean_init=0.1,
                 weight_rho_scale_init=0.1,
                 bias_mu_mean_init=0.1,
                 bias_mu_scale_init=0.1,
                 bias_rho_mean_init=0.1,
                 bias_rho_scale_init=0.1,
                 name='BNNRewardPredictor'):

        super().__init__()

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.name = name

        self.module = BayesianNN(
            input_dims=input_dim,
            output_dims=output_dim,
            weight_prior_scale=weight_prior_scale,
            bias_prior_scale=bias_prior_scale,
            hidden_sizes=hidden_sizes,
            prior_mix=prior_mix,
            empirical_complexity_loss=True,
        )


        # parameter initialization
        self.module.reset_parameters(
             weight_mu_mean=weight_mu_mean_init,
             weight_mu_scale=weight_mu_scale_init,
             weight_rho_mean=weight_rho_mean_init,
             weight_rho_scale=weight_rho_scale_init,
             bias_mu_mean=bias_mu_mean_init,
             bias_mu_scale=bias_mu_scale_init,
             bias_rho_mean=bias_rho_mean_init,
             bias_rho_scale=bias_rho_scale_init,
        )

    def sampled_cross_entropies(self, input, target, reduction='mean'):
        sampled_cross_entropies = []
        for sample in input:
            sampled_cross_entropies.append(
                nn.functional.cross_entropy(sample, target, reduction=reduction)
            )

        sampled_cross_entropies = torch.stack(sampled_cross_entropies)

        return sampled_cross_entropies

    def compute_preference_loss(self,
                                left_segs,
                                right_segs,
                                prefs,
                                dataset_size,
                                samples=1):

        if not left_segs.shape == right_segs.shape:
            raise ValueError('Left and Right segs should have the same shape')

        batch_size, segment_length, obs_dim = left_segs.shape

        inp = (torch.cat([left_segs, right_segs])
                    .reshape(2 * batch_size * segment_length, obs_dim))

        output = (self.module(inp, samples)
                      .reshape(samples, 2*batch_size, segment_length)
                      .sum(dim=-1))

        logits = output.reshape(samples, 2, batch_size).transpose(1, 2)

        complexity_cost = self.module.complexity_cost()
        likelihood_cost = self.sampled_cross_entropies(
            logits, prefs, reduction='mean')
        sampled_losses = (
            (1 / (dataset_size)) * complexity_cost + likelihood_cost
        )
        return torch.mean(sampled_losses)

    def propagate_preference_loss(self,
                                  left_segs,
                                  right_segs,
                                  prefs,
                                  dataset_size,
                                  samples=1):
        r"""Compute mean value of loss.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        if not left_segs.shape == right_segs.shape:
            raise ValueError('Left and Right segs should have the same shape')

        batch_size, segment_length, obs_dim = left_segs.shape

        inp = (torch.cat([left_segs, right_segs])
                    .reshape(2 * batch_size * segment_length, obs_dim))

        output = (self.module(inp, samples)
                      .reshape(samples, 2*batch_size, segment_length)
                      .sum(dim=-1))

        logits = output.reshape(samples, 2, batch_size).transpose(1, 2)

        complexity_cost = self.module.complexity_cost()
        likelihood_cost = self.sampled_cross_entropies(
            logits, prefs, reduction='mean')
        sampled_losses = (
            (1 / (dataset_size)) * complexity_cost + likelihood_cost
        )
        self.module.propagate_loss(sampled_losses)
        breakpoint()

        return torch.mean(sampled_losses)

    # pylint: disable=arguments-differ
    def forward(self, obs, samples=1):
        r"""Predict value based on paths.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.
        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.
        """

        logits = self.module(obs, samples)
        return logits.mean(dim=0)
