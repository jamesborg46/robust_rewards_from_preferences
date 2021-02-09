from utils import update_remote_agent_device
from garage.sampler.env_update import EnvUpdate
import os
import os.path as osp
import pickle
import wandb


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


class CloseRenderer(EnvUpdate):

    def __call__(self, old_env):
        old_env.close_renderer()
        return old_env


def log_episodes(trainer,
                 policy,
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
        agent_update=update_remote_agent_device(policy)
    )

    episodes = trainer._eval_sampler.obtain_exact_episodes(
        n_eps_per_worker=n_eps_per_worker,
        agent_update=update_remote_agent_device(policy)
    )

    env_updates = [CloseRenderer() for _ in range(trainer._eval_n_workers)]

    trainer._eval_sampler._update_workers(
        env_update=env_updates,
        agent_update=update_remote_agent_device(policy)
    )

    if capture_state:
        fname = osp.join(trainer._snapshotter.snapshot_dir,
                         'episode_logs',
                         f'episode_{trainer.step_itr}.pkl')

        if not osp.isdir(osp.dirname(fname)):
            os.makedirs(osp.dirname(fname))

        with open(fname, 'wb') as f:
            pickle.dump(episodes, f)

    if enable_render:
        for episode in episodes.split():
            video_file = episode.env_infos['video_filename'][0]
            assert '.mp4' in video_file
            wandb.log({
                os.path.basename(video_file): wandb.Video(video_file),
            }, step=trainer.step_itr)
