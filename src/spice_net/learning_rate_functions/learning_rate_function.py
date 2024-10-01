from abc import abstractmethod


class LearningRateFunction:

    @abstractmethod
    def call(self, iteration: int) -> float:
        pass
