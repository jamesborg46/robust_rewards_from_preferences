"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO
from garage import EpisodeBatch
from dowel import tabular
import numpy as np
from utils import corrcoef, update_remote_agent_device
from garage.sampler.env_update import EnvUpdate


class EnvConfigUpdate(EnvUpdate):

    def __init__(self,
                 capture_state=False,
                 enable_render=False,
                 file_prefix=""):

        self.capture_state = capture_state
        self.enable_render = enable_render
        self.file_prefix = file_prefix

    def __call__(self, old_env):
        old_env.set_capture_state(self.capture_state)
        old_env.enable_rendering(self.enable_render,
                                 file_prefix=self.file_prefix)
        return old_env


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

    def _log_episodes(self, trainer, capture_state=True, enable_render=False):

        env_updates = []
        for i in range(trainer._eval_n_workers):
            env_updates.append(EnvConfigUpdate(
                capture_state=capture_state,
                enable_render=enable_render,
                file_prefix=f"epoch_{trainer.step_itr:04}_worker_{i:02}_"
            ))

        trainer._eval_sampler._update_workers(
            env_update=env_updates
        )

        episodes = trainer._eval_sampler.obtain_exact_episodes(
            n_eps_per_worker=,
            agent_update=update_remote_agent_device(self.policy)
        )


