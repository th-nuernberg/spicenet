from .learning_rate_function import LearningRateFunction


class LinearLRF(LearningRateFunction):

    def __init__(self, slope: float, bias: float):
        self.__slope = slope
        self.__bias = bias

    def call(self, iteration: int) -> float:
        return self.__slope * iteration + self.__bias


def declining_linear_lrf_from_interval(start: float, stop: float, steps: int) -> LinearLRF:
    bias = start
    slope = (start - stop) / steps
    return LinearLRF(bias, slope)
