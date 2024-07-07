from .learning_rate_function import LearningRateFunction


class ConstLRF(LearningRateFunction):
    def __init__(self, value: float):
        self.__value = value

    def call(self, iteration: int) -> float:
        return self.__value
