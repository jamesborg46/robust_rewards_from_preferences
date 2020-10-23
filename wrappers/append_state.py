"""A wrapper for appending state to info for safety envs for gym.Env"""
import gym
from mujoco_py import MjSim, load_model_from_xml


class SafetyEnvStateAppender(gym.Wrapper):
    def reset(self):
        obs = self.env.reset()
        self.model_xml = self.model.get_xml()
        return obs

    def step(self, action):
        state = self.env.world.sim.get_state()
        obs, reward, done, info = self.env.step(action)
        info['state'] = state.flatten()
        info['model_xml'] = self.model_xml
        if self.model_xml is not None:
            self.model_xml = None
        return obs, reward, done, info

    def load_model(self, model_xml):
        self.env.reset()
        self.env.world.sim = MjSim(load_model_from_xml(model_xml))

    def render_state(self, state):
        self.env.world.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        self.env.render_lidar_markers = False
        rgb_array = self.render('rgb_array')
        return rgb_array
