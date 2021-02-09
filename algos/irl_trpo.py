"""Learning reward from preferences with TRPO"""

from dowel import tabular
from garage.torch.algos import TRPO
from garage import EpisodeBatch
import numpy as np
from utils import corrcoef, log_episodes


class IrlTRPO(TRPO):

    def __init__(self,
                 env_spec,
                 reward_predictor,
                 render_freq=200,
                 **kwargs,
                 ):

        super().__init__(env_spec=env_spec,
                         **kwargs)
        self._env_spec = env_spec
        self._reward_predictor = reward_predictor
        self._render_freq = render_freq

    def train(self, trainer):
        self._reward_predictor.pretrain(trainer)
        last_return = None

        for epoch in trainer.step_epochs():
            for _ in range(self._n_samples):
                trainer.step_path = trainer.obtain_samples(trainer.step_itr)
                last_return = self._train_once(trainer.step_itr,
                                               trainer.step_path)
                trainer.step_itr += 1

            if epoch and epoch % self._render_freq == 0:
                log_episodes(trainer, self.policy, enable_render=True)

        return last_return

    def _train_once(self, itr, paths):
        self._reward_predictor.predict_rewards(itr, paths)
        with tabular.prefix('ForwardAlgorithm/'):
            last_return = super()._train_once(itr, paths)
        self._reward_predictor.train_once(itr, paths)

        self._log_performance(paths)
        return last_return

    def _log_performance(self, paths):
        undiscounted_returns = []
        corrs = []
        for eps in EpisodeBatch.from_list(self._env_spec, paths).split():
            undiscounted_returns.append(sum(eps.env_infos['gt_reward']))
            corrs.append(corrcoef(eps.rewards, eps.env_infos['gt_reward']))

        with tabular.prefix('RewardPredictor/'):
            tabular.record('AverageGTReturn', np.mean(undiscounted_returns))
            tabular.record('StdGTReturn', np.std(undiscounted_returns))
            tabular.record('MaxGTReturn', np.max(undiscounted_returns))
            tabular.record('MinGTReturn', np.min(undiscounted_returns))
            tabular.record('AverageEpisodeRewardCorrelation',
                           np.mean(corrs))
