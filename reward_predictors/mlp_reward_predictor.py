"""MLP Reward Predictor"""
import torch
from torch import nn

from garage.torch.modules import MLPModule


class MLPRewardPredictor(nn.Module):
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
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='MLPRewardPredictor'):

        super().__init__()

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

    def compute_preference_loss(self, left_segs, right_segs, prefs, device='cpu'):
        if not left_segs.shape == right_segs.shape:
            raise ValueError('Left and Right segs should have the same shape')

        batch_size, segment_length, obs_dim = left_segs.shape

        inp = (torch.cat([left_segs, right_segs])
                    .reshape(2 * batch_size * segment_length, obs_dim)
                    .to(device))


        output = (self.module(inp)
                      .reshape(2*batch_size, segment_length)
                      .sum(dim=1))

        logits = output.reshape(2, batch_size).transpose(0, 1)

        loss = nn.functional.cross_entropy(logits, prefs)
        return loss

    def propagate_preference_loss(self, left_segs, right_segs, prefs, device='cpu'):
        r"""Compute mean value of loss.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        loss = self.compute_preference_loss(left_segs, right_segs, prefs, device)
        loss.backward()
        return loss

    def propagate_ranking_loss(self, segs, ranks):
        preds = self.module(segs)
        loss = nn.functional.mse_loss(preds, ranks.reshape(-1,1))
        loss.backward()
        return loss

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
