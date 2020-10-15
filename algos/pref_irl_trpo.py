"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO
from garage.torch import pad_to_last, filter_valids
from garage.np import discount_cumsum
from garage import StepType, EpisodeBatch
from garage.torch.optimizers import OptimizerWrapper
from dowel import tabular

import torch
import numpy as np


class PreferenceTRPO(TRPO):
    """Learning reward from preferences with TRPO

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 reward_predictor,
                 comparison_buffer,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 reward_predictor_optimizer=None,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy'):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

        self._reward_predictor = reward_predictor
        self.comparison_buffer = comparison_buffer

        if reward_predictor_optimizer:
            self._reward_predictor_optimizer = reward_predictor_optimizer
        else:
            self._reward_predictor_optimizer = OptimizerWrapper(
                torch.optim.Adam,
                self._reward_predictor)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.
        Args:
            trainer (LocalRunner): LocalRunner is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.
        Returns:
            float: The average return in last epoch cycle.
        """
        last_return = None

        self.pretrain_reward_predictor(trainer)

        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr)
                self.comparison_buffer.add_episode_batch(trainer.step_itr, eps)
                while self.comparison_buffer.requires_more_comparisons(trainer.step_itr):
                    self.comparison_buffer.sample_comparison()
                    self.comparison_buffer.label_unlabeled_comparisons()
                trainer.step_path = self._predict_rewards(eps.to_list())
                last_return = self._train_once(trainer.step_itr,
                                               trainer.step_path)
                trainer.step_itr += 1

        return last_return

    def _train(self,
               obs,
               actions,
               rewards,
               returns,
               advs,
               left_segs,
               right_segs,
               prefs):
        r"""Train the policy and value function with minibatch.
        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.
        """
        for dataset in self._reward_predictor_optimizer.get_minibatch(
                left_segs, right_segs, prefs):
            self._train_reward_predictor(*dataset)
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

    def _train_once(self, itr, paths):
        """Train the algorithm once.
        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.
        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.
        """
        obs, actions, rewards, returns, valids, baselines = \
            self._process_samples(paths)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        labeled_comparisons = (
            self.comparison_buffer.labeled_decisive_comparisons
        )

        left_segs = torch.tensor(
            [comp['left']['observations'] for comp in labeled_comparisons]
        ).type(torch.float32)

        right_segs = torch.tensor(
            [comp['right']['observations'] for comp in labeled_comparisons]
        ).type(torch.float32)

        preferences = torch.tensor(
            [comp['label'] for comp in labeled_comparisons]
        ).type(torch.long)

        with torch.no_grad():
            reward_predictor_loss_before = (
                self._reward_predictor.compute_preference_loss(left_segs,
                                                               right_segs,
                                                               preferences)
            )
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_before = self._compute_kl_constraint(obs)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat, left_segs, right_segs, preferences)

        with torch.no_grad():
            reward_predictor_loss_after = (
                self._reward_predictor.compute_preference_loss(left_segs,
                                                               right_segs,
                                                               preferences)
            )
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/KLBefore', kl_before.item())
            tabular.record('/KL', kl_after.item())
            tabular.record('/Entropy', policy_entropy.mean().item())

        with tabular.prefix(self._value_function.name):
            tabular.record('/LossBefore', vf_loss_before.item())
            tabular.record('/LossAfter', vf_loss_after.item())
            tabular.record('/dLoss',
                           vf_loss_before.item() - vf_loss_after.item())

        with tabular.prefix(self._reward_predictor.name):
            tabular.record('/LossBefore', reward_predictor_loss_before.item())
            tabular.record('/LossAfter', reward_predictor_loss_after.item())
            tabular.record('/dLoss',
                           reward_predictor_loss_before.item() -
                           reward_predictor_loss_after.item())

        tabular.record('BufferSize',
                       self.comparison_buffer.n_transitions_stored)

        tabular.record('NumberOfComparisons',
                       self.comparison_buffer.num_comparisons)

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = self._log_performance(
            itr,
            EpisodeBatch.from_list(
                self._env_spec, paths),
            discount=self._discount
        )
        return np.mean(undiscounted_returns)

    def _train_reward_predictor(self, left_segs, right_segs, prefs):
        r"""Train the reward predictor.
        Args:
            left_segs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            right_segs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            prefs (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).
        """
        self._reward_predictor_optimizer.zero_grad()
        loss = self._reward_predictor.compute_preference_loss(left_segs,
                                                              right_segs,
                                                              prefs)
        loss.backward()
        self._reward_predictor_optimizer.step()

        return loss

    def pretrain_reward_predictor(self, trainer):
        while (self.comparison_buffer.n_transitions_stored <
               (self.comparison_buffer._capacity * 0.8)):
            eps = trainer.obtain_episodes(0)
            self.comparison_buffer.add_episode_batch(0, eps)

        while self.comparison_buffer.requires_more_comparisons(0):
            self.comparison_buffer.sample_comparison()
            self.comparison_buffer.label_unlabeled_comparisons()

        labeled_comparisons = (
            self.comparison_buffer.labeled_decisive_comparisons
        )

        left_segs = torch.tensor(
            [comp['left']['observations'] for comp in labeled_comparisons]
        ).type(torch.float32)

        right_segs = torch.tensor(
            [comp['right']['observations'] for comp in labeled_comparisons]
        ).type(torch.float32)

        preferences = torch.tensor(
            [comp['label'] for comp in labeled_comparisons]
        ).type(torch.long)

        for _ in range(200):
            for dataset in self._reward_predictor_optimizer.get_minibatch(
                    left_segs, right_segs, preferences):
                self._train_reward_predictor(*dataset)

    def _predict_rewards(self, paths):
        """Predicts rewards using reward predictor and attaches predictions
        to paths
        """

        obs = torch.tensor(
            np.concatenate([path['observations'] for path in paths])
        ).type(torch.float32)

        with torch.no_grad():
            predicted_rewards = self._reward_predictor(obs).flatten().numpy()

        start = 0
        for path in paths:
            N = len(path['observations'])
            path['predicted_rewards'] = predicted_rewards[start:start+N]
            # path['predicted_rewards'] = path['env_infos']['gt_reward']
            start += N

        return paths

    def _process_samples(self, paths):
        r"""Process sample data based on the collected paths.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            paths (list[dict]): A list of collected paths
        Returns:
            torch.Tensor: The observations of the environment
                with shape :math:`(N, P, O*)`.
            torch.Tensor: The actions fed to the environment
                with shape :math:`(N, P, A*)`.
            torch.Tensor: The acquired rewards with shape :math:`(N, P)`.
            list[int]: Numbers of valid steps in each paths.
            torch.Tensor: Value function estimation at each step
                with shape :math:`(N, P)`.
        """
        valids = torch.Tensor([len(path['actions']) for path in paths]).int()
        obs = torch.stack([
            pad_to_last(path['observations'],
                        total_length=self.max_episode_length,
                        axis=0) for path in paths
        ])
        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=self.max_episode_length,
                        axis=0) for path in paths
        ])
        predicted_rewards = torch.stack([
            pad_to_last(path['predicted_rewards'],
                        total_length=self.max_episode_length)
            for path in paths
        ])
        returns = torch.stack([
            pad_to_last(
                discount_cumsum(path['predicted_rewards'],
                                self.discount).copy(),
                total_length=self.max_episode_length) for path in paths
        ])
        with torch.no_grad():
            baselines = self._value_function(obs)

        return obs, actions, predicted_rewards, returns, valids, baselines

    def _log_performance(self, itr, batch, discount, prefix='Evaluation'):
        """Evaluate the performance of an algorithm on a batch of episodes.
        Args:
            itr (int): Iteration number.
            batch (EpisodeBatch): The episodes to evaluate with.
            discount (float): Discount value, from algorithm's property.
            prefix (str): Prefix to add to all logged keys.
        Returns:
            numpy.ndarray: Undiscounted returns.
        """
        returns = []
        undiscounted_returns = []
        termination = []
        success = []
        for eps in batch.split():
            returns.append(discount_cumsum(eps.env_infos['gt_reward'],
                                           discount))
            undiscounted_returns.append(sum(eps.env_infos['gt_reward']))
            termination.append(
                float(
                    any(step_type == StepType.TERMINAL
                        for step_type in eps.step_types)))
            if 'success' in eps.env_infos:
                success.append(float(eps.env_infos['success'].any()))

        average_discounted_return = np.mean([rtn[0] for rtn in returns])

        with tabular.prefix(prefix + '/'):
            tabular.record('Iteration', itr)
            tabular.record('NumEpisodes', len(returns))

            tabular.record('AverageDiscountedGTReturn',
                           average_discounted_return)
            tabular.record('AverageGTReturn', np.mean(undiscounted_returns))
            tabular.record('StdGTReturn', np.std(undiscounted_returns))
            tabular.record('MaxGTReturn', np.max(undiscounted_returns))
            tabular.record('MinGTReturn', np.min(undiscounted_returns))
            tabular.record('TerminationRate', np.mean(termination))
            if success:
                tabular.record('SuccessRate', np.mean(success))

        return undiscounted_returns
