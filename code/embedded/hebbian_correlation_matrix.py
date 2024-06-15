import numpy as np

from spice_net_som import SpiceNetSom


class HebbianCorrelationMatrix:

    def __init__(self, som_1: SpiceNetSom, som_2: SpiceNetSom):
        self.__weights = np.ones((len(som_1), len(som_2)))
        self.__activation_bar_1 = np.zeros(len(som_1))
        self.__activation_bar_2 = np.zeros(len(som_2))
        self.__som_1 = som_1
        self.__som_2 = som_2

    def get_matrix(self) -> np.ndarray:
        """
        Get the weight matrix.
        :return: The matrix dimension 1 is the first som given to the constructor.
        """
        return self.__weights

    def fit(self, values_som_1: list[float], values_som_2: list[float], iteration: int):
        learning_rate_weights = 0.7
        learning_rate_represented_value = 0.7
        k = float(iteration)
        for i in range(len(values_som_1)):
            k += 1.0
            # TODO: Verlierer müssen abnehmen!
            winner_som_1, activation_1 = self.__som_1.get_winning_neuron_index(values_som_1[i])
            winner_som_2, activation_2 = self.__som_2.get_winning_neuron_index(values_som_2[i])

            self.__activation_bar_1[winner_som_1] = ((1.0 - learning_rate_represented_value) * self.__activation_bar_1[winner_som_1]
                                                     * (k - 1.0) + learning_rate_represented_value * activation_1)
            self.__activation_bar_2[winner_som_2] = ((1.0 - learning_rate_represented_value) * self.__activation_bar_2[winner_som_2]
                                                     * (k - 1.0) + learning_rate_represented_value * activation_2)

            weights_delta = (
                    learning_rate_weights * (activation_1 - self.__activation_bar_1[winner_som_1])
                    * (activation_2 - self.__activation_bar_2[winner_som_2])
            )
            self.__weights[winner_som_1, winner_som_2] += weights_delta
