import math

from .learning_rate_function import LearningRateFunction


class ExpEDecayLRF(LearningRateFunction):
    def __init__(self, speed: float, approached_value: float, x_shift: float = 0.0):
        """
        (e^(-iteration * speed)) - bias
        :param speed:
        :param approached_value:
        """
        self.__bias: float = approached_value
        self.__speed: float = speed
        self.__x_shift: float = x_shift

    def call(self, iteration: int) -> float:
        return math.exp(-iteration * self.__speed + self.__x_shift) - self.__bias
