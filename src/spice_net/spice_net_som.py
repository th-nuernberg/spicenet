import math
from typing import Callable

import numpy as np

from .learning_rate_functions import LearningRateFunction


def local_min(interval_start: float, interval_end: float, eps, t, function: Callable[[float], float]):
    c = 0.381966  # c = (3 - sqrt(5)) / 2
    v = w = x = interval_start + c * (interval_end - interval_start)
    e = 0
    fv = fw = fx = function(x)

    while True:
        m = 0.5 * (interval_start + interval_end)
        tol = eps * abs(x) + t
        t2 = 2 * tol
        if abs(x - m) > t2 - 0.5 * (interval_end - interval_start):
            p = q = r = 0
            if abs(e) > tol:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0:
                    p = -p
                else:
                    q = -q
                r = e
                e = d
            if abs(p) < abs(0.5 * q * r) and p < q * (interval_start - x) and p < q * (interval_end - x):
                d = p / q
                u = x + d
                if u - interval_start < t2 or interval_end - u < t2:
                    d = tol if x < m else -tol
            else:
                e = (interval_end if x < m else interval_start) - x
                d = c * e
            u = x + (d if abs(d) >= tol else (tol if d > 0 else -tol))
            fu = function(u)
            if fu <= fx:
                if u < x:
                    interval_end = x
                else:
                    interval_start = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    interval_start = u
                else:
                    interval_end = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
        else:
            break
    return fx


class SpiceNetSom:
    """
    This self organizing map implementation is specifically for SpiceNet. It creates a 1D som.
    For more information read the theory papers included.
    """

    def __init__(self,
                 n_neurons: int,
                 value_range_start: float,
                 value_range_end: float,
                 lrf_tuning_curve: LearningRateFunction,
                 lrf_interaction_kernel: LearningRateFunction,
                 make_2d_input: bool = False,):
        """
        Creates a SpiceNetSom with neurons equally distributed across the specified value range.
        :param n_neurons: The number of neurons that should be created.
        :param value_range_start: Start of the believed value range. (This is only relevant for the initialisation.
        It is possible to fit values outside of this range!)
        :param value_range_end: End of the believed value range. (This is only relevant for the initialisation.
        It is possible to fit values outside of this range!)
        :param lrf_tuning_curve: Use this parameter to define how fast the tuning curve is changed.
        :param lrf_interaction_kernel: Use this parameter to define how strong the interaction kernel affects the preferred value of a node.
        """
        self.__lrf_interaction_kernel = lrf_interaction_kernel
        self.__lrf_tuning_curve = lrf_tuning_curve
        self.__iteration = 0

        distance = value_range_end - value_range_start
        step_size = distance / n_neurons

        self.__neurons = []
        pos = value_range_start + step_size / 2.0

        for i in range(n_neurons):
            if make_2d_input:
                new_neuron = SpiceNetSom.__SomNeuron(np.array([pos, pos]))
            else:
                new_neuron = SpiceNetSom.__SomNeuron(pos)
            self.__neurons.append(new_neuron)
            pos += step_size

    def __len__(self) -> int:
        return len(self.__neurons)

    def print_neurons(self):
        for neuron in self.__neurons:
            print(neuron)

    def fit(self, values: list[float | np.ndarray[any, np.dtype[np.float64]]], epochs: int):
        for epoch in range(epochs):
            for i in range(len(values)):
                winning_neuron_index, _ = self.__argmax_neuron_activation(values[i])

                for j in range(len(self.__neurons)):
                    self.__neurons[j].update(values[i],
                                             self.__lrf_tuning_curve.call(self.__iteration),
                                             self.__lrf_interaction_kernel.call(self.__iteration),
                                             j - winning_neuron_index)
                self.__iteration += 1

    def get_as_matrix(self) -> np.ndarray:
        """
        Returns all neurons like a table with column 0 representing the preferred value
        and column 1 representing the tuning curve width.
        :return: A numpy matrix.
        """
        return np.array([[neuron.preferred_value for neuron in self.__neurons],
                         [neuron.tuning_curve_width for neuron in self.__neurons]]).transpose()

    def get_activation_vector(self, value: float | np.ndarray[any, np.dtype[np.float64]]) -> np.array:
        return np.array([neuron.activation_for_value(value) for neuron in self.__neurons])

    def calculate_activation_values(self, values: list[float | np.ndarray[any, np.dtype[np.float64]]]):
        """
        Calculates activation values for all neurons in the som.
        :param values:
        :return: A numpy array the first col is the preferred value, second col is the tuning curve width
        and the following cols are the activation values.
        """
        activation_values = np.array([neuron.activation_for_values(values) for neuron in self.__neurons])
        return np.concatenate((
            np.array(
                [[neuron.preferred_value for neuron in self.__neurons],
                 [neuron.tuning_curve_width for neuron in self.__neurons]]).transpose(),
            activation_values),
            axis=1)

    def get_winning_neuron_index(self, value: float | np.ndarray[any, np.dtype[np.float64]]) -> (int, float):
        winning_neuron_index, activation_values = self.__argmax_neuron_activation(value)
        return winning_neuron_index, activation_values[winning_neuron_index]

    def naive_decode(self, value: float, neuron_index: int) -> float:
        # activation_value = self.__neurons[neuron_index].activation_for_value(value)
        activation_value = value
        # activation_value += 1000000
        # print(f'neuron: {neuron_index} value: {value} activation: {activation_value}')
        # print(f'prefered_value: {self.__neurons[neuron_index].preferred_value}')
        # print(f'width: {self.__neurons[neuron_index].tuning_curve_width}')
        # print(self.__neurons[neuron_index].tuning_curve_width)
        # print(math.sqrt(2 * math.pi) * activation_value * self.__neurons[neuron_index].tuning_curve_width ** 2)
        # print(2 * self.__neurons[neuron_index].tuning_curve_width ** 2 * math.log(
        #     math.sqrt(2 * math.pi) * activation_value * self.__neurons[neuron_index].tuning_curve_width ** 2))
        r = math.sqrt(2 * self.__neurons[neuron_index].tuning_curve_width ** 2 * math.log(
            math.sqrt(2 * math.pi) * activation_value * self.__neurons[neuron_index].tuning_curve_width ** 2, 10))

        if neuron_index < len(self.__neurons) / 2:
            return self.__neurons[neuron_index].preferred_value - r
        else:
            return self.__neurons[neuron_index].preferred_value + r

    def decode(self, activations: np.array) -> float:
        # input is given
        # cost = som_known(input) - (som_searched(x) * correlation)
        # better:
        # cost = som_known_winning(input) - (som_searched_winning(x) * correlation)
        # tol=(1.0e-6)*(limL + limH)/2.0;
        pass

    def __argmax_neuron_activation(self, value: float | np.ndarray[any, np.dtype[np.float64]]):
        """
        Calculates the neuron with the highest activation value.
        :param value: The value for wich the activation values should be calculated.
        :return: The index of the winning neuron
        and a dictionary containing the index of a neuron with the calculated activation value.
        """
        winning_neuron_index: int = 0
        max_activation = self.__neurons[0].activation_for_value(value)
        activation_dict = {0: max_activation}

        for i in range(1, len(self.__neurons)):
            new_activation = self.__neurons[i].activation_for_value(value)
            # add new value to a dictionary so the activations only have to be calculated a single time
            activation_dict[i] = new_activation

            if new_activation > max_activation:
                max_activation = new_activation
                winning_neuron_index = i
        return winning_neuron_index, activation_dict

    class __SomNeuron:
        """
        This object represents a som neuron. Since this som is 1D the neuron only knows th previous and next neuron.
        """

        def __init__(self,
                     preferred_value: float | np.ndarray[any, np.dtype[np.float64]],
                     tuning_curve_width: float = 0.001):
            """
            Creates a som neuron.
            :param preferred_value: The "expected value" of the Normal distribution
            representing the activation function.
            :param tuning_curve_width: The width of the tuning curve (the Normal distribution).
            """
            self.preferred_value = preferred_value
            """ The "expected value" of the Normal distribution representing the activation function. """
            self.tuning_curve_width = tuning_curve_width
            """ The width of the tuning curve (the Normal distribution). """

        def __str__(self):
            return f'SomeNeuron: preferred_value={self.preferred_value}, tuning_curve_width={self.tuning_curve_width}'

        def activation_for_value(self, value: float | np.ndarray[any, np.dtype[np.float64]]):
            """
            Returns the activation value of the neuron.
            :param value: The value for wich the activation has to be calculated.
            :return: The "height" of the tuning curve at the position of the value.
            """
            if isinstance(value, float):
                return (
                        (1.0 / (math.sqrt(2.0 * math.pi) * self.tuning_curve_width))
                        *
                        math.exp(
                            (-(value - self.preferred_value) ** 2) /
                            (2.0 * self.tuning_curve_width ** 2))
                )
            else:
                return (
                        (1.0 / (math.sqrt(2.0 * math.pi) * self.tuning_curve_width))
                        *
                        math.exp(
                            (-(np.linalg.vector_norm(value - self.preferred_value)) ** 2) /
                            (2.0 * self.tuning_curve_width ** 2))
                )

        def activation_for_values(self, values: list[float | np.ndarray[any, np.dtype[np.float64]]]):
            """
            Returns the activation values for a list of values.
            :param values: The values for wich the activation has to be calculated.
            :return: An array of activation values, ordered like the input list.
            """
            return np.array([self.activation_for_value(value) for value in values])

        def update(self, value: float | np.ndarray[any, np.dtype[np.float64]], learn_rate: float, interaction_kernel_learning_rate: float,
                   distance_to_winner: int):
            """
            Updates the weights of the neurons.
            :param value: The value in wich "direction" the neuron should move.
            :param learn_rate: The learn rate. (This value should be depending on the iteration.)
            :param interaction_kernel_learning_rate: The sigma of the interaction kernel.
            (This value should be depending on the iteration.)
            :param distance_to_winner: The distance between this neuron and the winning neuron, in the SOM.
            Here are not the weights / postions in the value range relevant.
            (N1 weight: 31.9) --- (N2 weight: 32) --- (N3 weight: 32.4) --- (N4 weight: 33)
            The Distance of N4 to N1 is 3
            :return:
            """
            interaction_kernel_value = math.exp(
                (-abs(distance_to_winner) ** 2) / (2 * interaction_kernel_learning_rate ** 2))
            self.preferred_value += learn_rate * interaction_kernel_value * (value - self.preferred_value)
            if isinstance(value, float):
                self.tuning_curve_width += learn_rate * interaction_kernel_value * (
                        (value - self.preferred_value) ** 2 - self.tuning_curve_width ** 2
                )
            else:
                self.tuning_curve_width += learn_rate * interaction_kernel_value * (
                        np.linalg.vector_norm(value - self.preferred_value) ** 2 - self.tuning_curve_width ** 2
                )

