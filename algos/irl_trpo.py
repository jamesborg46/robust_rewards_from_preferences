"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO
from garage import EpisodeBatch
from dowel import tabular
import numpy as np
from utils import corrcoef


class IrlTRPO(TRPO):

    def __init__(self,
                 env_spec,
                 reward_predictor,
                 val_freq=100,
                 **kwargs,
                 ):

        super().__init__(env_spec=env_spec, **kwargs)
        self._env_spec = env_spec
        self._reward_predictor = reward_predictor
        self._val_freq = val_freq

    def train(self, trainer):
        self._reward_predictor.pretrain(trainer)
        super().train(trainer)

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

