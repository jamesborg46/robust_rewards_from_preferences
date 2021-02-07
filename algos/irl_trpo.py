"""Learning reward from preferences with TRPO"""

from dowel import tabular
from garage.sampler.env_update import EnvUpdate
from garage.torch.algos import TRPO
from garage import EpisodeBatch
import numpy as np
import os
import os.path as osp
import pickle
from utils import corrcoef, update_remote_agent_device


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
                 render_freq=2,
                 **kwargs,
                 ):

        super().__init__(env_spec=env_spec, **kwargs)
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
                self._log_episodes(trainer, enable_render=True)

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

    def _log_episodes(self,
                      trainer,
                      n_eps_per_worker=1,
                      capture_state=True,
                      enable_render=False):

        env_updates = []
        for i in range(trainer._eval_n_workers):
            env_updates.append(EnvConfigUpdate(
                capture_state=capture_state,
                enable_render=enable_render,
                file_prefix=f"epoch_{trainer.step_itr:04}_worker_{i:02}_"
            ))

        trainer._eval_sampler._update_workers(
            env_update=env_updates,
            agent_update=update_remote_agent_device(self.policy)
        )

        episodes = trainer._eval_sampler.obtain_exact_episodes(
            n_eps_per_worker=n_eps_per_worker,
            agent_update=update_remote_agent_device(self.policy)
        )

        if capture_state:
            fname = osp.join(trainer._snapshotter.snapshot_dir,
                             'episode_logs',
                             f'episode_{trainer.step_itr}.pkl')

            if not osp.isdir(osp.dirname(fname)):
                os.makedirs(osp.dirname(fname))

            with open(fname, 'wb') as f:
                pickle.dump(episodes, f)

