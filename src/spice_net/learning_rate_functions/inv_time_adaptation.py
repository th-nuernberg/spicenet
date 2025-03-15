from . import LearningRateFunction


class InverseTimeAdaptation(LearningRateFunction):

    def __init__(self,
                 init_value: float,
                 final_value: float,
                 planned_iterations: int,
                 iteration_0=1):
        self.__planned_iterations = planned_iterations
        self.__B = ((final_value * planned_iterations - init_value * iteration_0)
                  / init_value - final_value)
        self.__A = init_value * iteration_0 + self.__B * iteration_0

    def call(self, iteration: int) -> float:
        t = self.__planned_iterations - iteration
        return self.__A / (t + self.__B)

    def get_parameters(self) -> dict:
        # Could return either initial values or current A and B, we chose current A and B to stay consistent when exporting and reimporting for further training
        return {"A": self.__A, "B": self.__B, "planned_iterations": self.__planned_iterations}
    
    @classmethod
    def from_parameters(cls, A: float, B: float, planned_iterations: int) -> 'InverseTimeAdaptation':
        instance = cls()
        instance.__A = A
        instance.__B = B
        instance.__planned_iterations = planned_iterations
        return instance