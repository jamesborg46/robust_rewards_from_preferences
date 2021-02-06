"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO


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
        last_return = super()._train_once(itr, paths)
        self._reward_predictor.train_once(itr, paths)
        return last_return
