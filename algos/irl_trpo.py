"""Learning reward from preferences with TRPO"""

from dowel import tabular
from garage.torch.algos import TRPO
from utils import log_episodes, log_gt_performance


class IrlTRPO(TRPO):

    def __init__(self,
                 env_spec,
                 reward_predictor,
                 snapshot_dir,
                 log_sampler,
                 render_freq=200,
                 **kwargs,
                 ):

        super().__init__(env_spec=env_spec,
                         **kwargs)
        self._env_spec = env_spec
        self._reward_predictor = reward_predictor
        self._render_freq = render_freq
        self._snapshot_dir = snapshot_dir
        self._log_sampler = log_sampler

    def train(self, trainer):
        self._reward_predictor.pretrain(trainer)
        last_return = super().train(trainer)
        return last_return

    def _train_once(self, itr, eps):
        eps = self._reward_predictor.predict_rewards(itr, eps)
        with tabular.prefix('ForwardAlgorithm/'):
            last_return = super()._train_once(itr, eps)
        self._reward_predictor.train_once(itr, eps)

        log_gt_performance(itr, eps, self._discount)
        if itr and itr % self._render_freq == 0:
            log_episodes(itr,
                         self._snapshot_dir,
                         self._log_sampler,
                         self.policy,
                         enable_render=True)

        return last_return
