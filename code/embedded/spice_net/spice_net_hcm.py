import numpy as np

from .learning_rate_functions import LearningRateFunction
from .spice_net_som import SpiceNetSom


class SpiceNetHcm:

    def __init__(self,
                 som_1: SpiceNetSom,
                 som_2: SpiceNetSom,
                 lrf_weights: LearningRateFunction,
                 lrf_trust_of_new: LearningRateFunction):
        """
        This object represents a hebbian correlation matrix between 2 SOMs.
        :param som_1: The first SOM.
        :param som_2: The second SOM.
        :param lrf_weights: The learning rate of the weights, how strong each update affects the matrix.
        :param lrf_trust_of_new: This implementation uses the hebbian covariance learning mechanism. This uses an 'average value', this value is update continuous in this implementation. Use this parameter to tell how much you trust that the new value and want it to change the average. For more information on the mechanism look here: https://rkypragada.medium.com/hebbian-learning-c2166ac0f48d
        """
        self.__lrf_trust_of_new = lrf_trust_of_new
        self.__lrf_weights = lrf_weights
        self.__weights = np.ones((len(som_1), len(som_2)))
        self.__activation_bar_vector_1: np.array = np.zeros(len(som_1))
        self.__activation_bar_vector_2: np.array = np.zeros(len(som_2))
        self.__som_1 = som_1
        self.__som_2 = som_2
        self.__iteration = 0

    def get_soms(self):
        return self.__som_1, self.__som_2

    def get_matrix(self) -> np.ndarray:
        """
        Get the weight matrix.
        :return: The matrix y-axis is the first som given to the constructor (x-axis the second one).
        """
        return self.__weights

    def calculate_som_1_to_2(self, a_array: np.array) -> np.array:
        if a_array.shape != (len(self.__som_1),):
            raise ValueError("The input vector must be of the same length as the som amount of neurons.")
        # Take each column of the weight matrix and calculate the dot product with the input array.
        return np.array([a_array.dot(self.__weights[:, i]) for i in range(self.__weights.shape[0])])

    def fit(self,
            values_som_1: list[float],
            values_som_2: list[float],
            epochs: int):
        """
        Fits the hebbian correlation matrix.

        :param values_som_1: The values of the first som.
        :param values_som_2: The values of the second som.
        :param epochs: How many times the matrix is fitted to the data.
        """
        if len(values_som_1) != len(values_som_2):
            raise Exception('The length of the values of the first som must be equal to the length of the second som.')

        for _ in range(epochs):
            for i in range(len(values_som_1)):
                activation_vector_som_1 = self.__som_1.get_activation_vector(values_som_1[i])
                activation_vector_som_2 = self.__som_2.get_activation_vector(values_som_2[i])
                self.__activation_bar_vector_1 = ((1.0 - self.__lrf_trust_of_new.call(self.__iteration))
                                                  * self.__activation_bar_vector_1
                                                  + self.__lrf_trust_of_new.call(self.__iteration)
                                                  * activation_vector_som_1)
                self.__activation_bar_vector_2 = ((1.0 - self.__lrf_trust_of_new.call(self.__iteration))
                                                  * self.__activation_bar_vector_2
                                                  + self.__lrf_trust_of_new.call(self.__iteration)
                                                  * activation_vector_som_2)

                # Som 1 represents the y-axis and Som 2 the x-axis
                weights_delta_matrix = (
                        self.__lrf_weights.call(self.__iteration)
                        * np.matrix(activation_vector_som_1 - self.__activation_bar_vector_1)
                        .transpose()
                        .dot(np.matrix(activation_vector_som_2 - self.__activation_bar_vector_2))
                )
                self.__weights += weights_delta_matrix
                self.__iteration += 1
