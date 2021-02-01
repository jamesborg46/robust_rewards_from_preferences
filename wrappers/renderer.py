"""A wrapper for appending state to info for safety envs for gym.Env"""
import gym
import os
from collections import defaultdict
from gym.wrappers.monitoring import video_recorder


class Renderer(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)

        self.video_enabled = False
        self.directory = directory
        self.video_recorder = None
        self.names = defaultdict(int)
        self.file_prefix = ""

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def enable_rendering(self, rendering_enabled, file_prefix=""):
        self.video_enabled = rendering_enabled
        self.file_prefix = file_prefix

        if not rendering_enabled and self.video_recorder:
            self.video_recorder.close()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        if self.video_enabled:
            self.video_recorder.capture_frame()

        return obs, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.video_enabled:
            self.reset_video_recorder()
        return observation

    def close(self):
        super().close()
        if self.video_recorder:
            self.video_recorder.close()

    def __del__(self):
        self.close()

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self.video_recorder.close()

        name = self.file_prefix
        skill = self.metadata.get('skill')
        if skill is not None:
            name = name + f'skill_{skill:02}'
        self.names[name] += 1

        ep_id = self.names[name]
        filename = name + f"_id_{ep_id:002}"

        # Start recording the next video.
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(
                self.directory,
                filename
            ),
            metadata=self.metadata,
            enabled=self.video_enabled,
        )
        self.video_recorder.capture_frame()

