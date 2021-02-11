from utils import update_remote_agent_device
from garage.sampler.env_update import EnvUpdate
import os
import os.path as osp
import pickle
import wandb
import ray


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


def log_episodes(itr,
                 snapshot_dir,
                 sampler,
                 policy,
                 n_eps_per_worker=1,
                 capture_state=True,
                 enable_render=False):

    n_workers = sampler._worker_factory.n_workers
    env_updates = []
    for i in range(n_workers):
        env_updates.append(EnvConfigUpdate(
            capture_state=capture_state,
            enable_render=enable_render,
            file_prefix=f"epoch_{itr:04}_worker_{i:02}_"
        ))

    sampler._update_workers(
        env_update=env_updates,
        agent_update=update_remote_agent_device(policy)
    )

    episodes = sampler.obtain_exact_episodes(
        n_eps_per_worker=n_eps_per_worker,
        agent_update=update_remote_agent_device(policy)
    )

    env_updates = [CloseRenderer() for _ in range(n_workers)]

    updates = sampler._update_workers(
        env_update=env_updates,
        agent_update=update_remote_agent_device(policy)
    )

    while updates:
        ready, updates = ray.wait(updates)

    if capture_state:
        fname = osp.join(snapshot_dir,
                         'episode_logs',
                         f'episode_{itr}.pkl')

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
            }, step=itr)
