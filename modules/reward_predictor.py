import abc


class RewardPredictor(abc.ABC):

    @abc.abstractmethod
    def pretrain(self, trainer):
        pass

    @abc.abstractmethod
    def train_once(self, itr, paths):
        pass

    @abc.abstractmethod
    def predict_rewards(self, itr, paths):
        pass
