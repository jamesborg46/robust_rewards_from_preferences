"""Learning reward from preferences with TRPO"""

from garage.torch.algos import TRPO
from dowel import tabular


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
        return last_return

    # def _log_performance(self, itr, paths):
    #     returns = []
    #     undiscounted_returns = []
    #     termination = []
    #     success = []
    #     for eps in batch.split():
    #         returns.append(discount_cumsum(eps.rewards, discount))
    #         undiscounted_returns.append(sum(eps.rewards))
    #         termination.append(
    #             float(
    #                 any(step_type == StepType.TERMINAL
    #                     for step_type in eps.step_types)))
    #         if 'success' in eps.env_infos:
    #             success.append(float(eps.env_infos['success'].any()))

    #     average_discounted_return = np.mean([rtn[0] for rtn in returns])

    #     with tabular.prefix(prefix + '/'):
    #         tabular.record('Iteration', itr)
    #         tabular.record('NumEpisodes', len(returns))

    #         tabular.record('AverageDiscountedReturn', average_discounted_return)
    #         tabular.record('AverageReturn', np.mean(undiscounted_returns))
    #         tabular.record('StdReturn', np.std(undiscounted_returns))
    #         tabular.record('MaxReturn', np.max(undiscounted_returns))
    #         tabular.record('MinReturn', np.min(undiscounted_returns))
    #         tabular.record('TerminationRate', np.mean(termination))
    #         if success:
    #             tabular.record('SuccessRate', np.mean(success))

