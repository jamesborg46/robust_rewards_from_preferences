"""A wrapper for appending state to info for safety envs for gym.Env"""
import gym


class SafetyEnvStateAppender(gym.Wrapper):

    def step(self, action):
        state = self.env.world.sim.get_state()
        obs, reward, done, info = self.env.step(action)
        info['state'] = state.flatten()
        return obs, reward, done, info

    def render_state(self, state):
        if not hasattr(self.env.world, 'sim'):
            self.env.reset()
        breakpoint()
        old_state = self.env.world.sim.get_state().flatten()
        self.env.world.sim.set_state_from_flattened(state)
        rgb_array = self.render('rgb_array')
        self.env.worl.sim.set_state_from_flatenned(old_state)
        return rgb_array
