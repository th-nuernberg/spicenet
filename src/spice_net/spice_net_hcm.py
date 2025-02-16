import typing

import numpy as np

from .learning_rate_functions import LearningRateFunction
from .spice_net_som import SpiceNetSom


class SpiceNetHcm:

    def __init__(self,
                 soms: typing.Collection[SpiceNetSom],
                 lrf_weights: LearningRateFunction,
                 lrf_trust_of_new: LearningRateFunction):
        """
        This object represents a hebbian correlation matrix between 2 SOMs.
        :param soms: A collection of SpiceNetSom objects mapped to a string key.
        :param lrf_weights: The learning rate of the weights, how strong each update affects the matrix.
        :param lrf_trust_of_new: This implementation uses the hebbian covariance learning mechanism. This uses an 'average value', this value is update continuous in this implementation. Use this parameter to tell how much you trust that the new value and want it to change the average. For more information on the mechanism look here: https://rkypragada.medium.com/hebbian-learning-c2166ac0f48d
        """
        self.__lrf_trust_of_new = lrf_trust_of_new
        self.__lrf_weights = lrf_weights
        self.__soms: list[SpiceNetSom] = [som for som in soms]

        index = 0
        # self.__activation_bars: list[np.ndarray] = []
        self.__activation_bars: list[np.ndarray] = [np.zeros(len(som)) for som in soms]
        self.__current_activations: list[np.ndarray] = [np.zeros(len(som)) for som in soms]
        self.__index_to_som: dict[int, SpiceNetSom] = {}
        self.__som_key_to_index: dict[str, int] = {}
        for som in soms:
            self.__index_to_som[index] = som
            self.__index_to_som[som.get_key()] = index
            # self.__activation_bars.append(np.zeros(len(som)))
            self.__som_key_to_index[som.get_key()] = index
            index += 1
        self.__weights = np.ones([len(som) for som in soms])
        self.__iteration = 0

    def get_soms(self):
        return self.__soms

    def get_matrix(self) -> np.ndarray:
        """
        Get the weight matrix.
        :return: The matrix y-axis is the first som given to the constructor (x-axis the second one).
        """
        return self.__weights

    def calculate_should_pattern(self, target_som_key: str,
                                 input_activations: typing.Mapping[str, np.ndarray]) -> np.ndarray:
        """
        The anticipated activation pattern.
        :param target_som_key:
        :param input_activations:
        :return: This function returns a relative representation of the activation.
        These are not the numbers return by the action function, but the ratio of the activation values is close to the real value.
        """
        target_index = self.__som_key_to_index[target_som_key]
        sorted_vectors = []
        for som in self.__soms:
            if som.get_key() == target_som_key:
                continue
            if input_activations[som.get_key()] is None and som.get_key() != target_som_key:
                raise ValueError('Missing value for ' + som.get_key())
            else:
                sorted_vectors.append(input_activations[som.get_key()])

        result = []
        for i in range(len(self.__soms[target_index])):
            array_slice = np.take(self.__weights, indices=i, axis=target_index)
            grids = np.meshgrid(*sorted_vectors, indexing='ij')
            for grid in grids:
                array_slice *= grid
            result.append(np.sum(array_slice))
        return np.array(result)

        # TODO: change to relative collapse instead of absolute
        result = []
        for i in range(len(self.__soms[target_index])):
            index_list = [0 for _ in range(len(self.__soms))]
            index_list[target_index] = i
            keys = [key for key in input_activations.keys()]
            result.append(self.__should_help(keys, index_list, [], input_activations))
        return np.array(result)

    def __should_help(self, remaining_keys: list[str],
                      indices: list[int],
                      values: list[float],
                      input_activations: typing.Mapping[str, np.ndarray]) -> float:
        """

        :param remaining_keys:
        :param indices:
        :param values:
        :param input_activations:
        :return:
        """
        if len(remaining_keys) > 0:
            current_key = remaining_keys.pop()
            som_weight_index = self.__som_key_to_index[current_key]
            result = 0.0
            for i in range(len(input_activations[current_key])):
                val = input_activations[current_key][i]
                ind = indices.copy()
                ind[som_weight_index] = i
                result += self.__should_help(remaining_keys, ind, values + [val], input_activations)
            return result
        else:
            result = 1.0
            for value in values:
                result *= value
            return self.__weights[*indices] * result

    def fit(self,
            values: typing.Mapping[str, list[float | np.ndarray[any, np.dtype[np.float64]]]],
            epochs: int):
        """
        Fits the hebbian correlation matrix.

        :param values:
        :param epochs: How many times the matrix is fitted to the data.
        """

        data_len = -1
        for som in self.__soms:
            if values[som.get_key()] is None:
                raise ValueError('Missing value for ' + som.get_key())
            if data_len != -1 and data_len != len(values[som.get_key()]):
                raise ValueError('The length of the data columns is not allways the same.')
            if data_len == -1:
                data_len = len(values[som.get_key()])

        for _ in range(epochs):
            for i in range(data_len):
                l = []
                for som in self.__soms:
                    index_in_weight_matrix = self.__som_key_to_index[som.get_key()]
                    activation_vector = som.get_activation_vector(values[som.get_key()][i])
                    self.__current_activations[index_in_weight_matrix] = activation_vector
                    self.__activation_bars[index_in_weight_matrix] = (
                            (1.0 - self.__lrf_trust_of_new.call(self.__iteration))
                            * self.__activation_bars[index_in_weight_matrix]
                            + self.__lrf_trust_of_new.call(self.__iteration)
                            * activation_vector)
                    l.append(activation_vector - self.__activation_bars[index_in_weight_matrix])
                shape_of_weight = np.shape(self.__weights)
                # with np.nditer(self.__weights, op_flags=['readwrite'], flags=['multi_index']) as it:
                #    for x in it:
                #        x[...] += self.__calculate_delta_value(it.multi_index)
                #        #print("%d <%s>" % (x, it.multi_index))

                delta_matrix = self.__generate_n_dimensional_tensor(l)
                self.__weights += delta_matrix * self.__lrf_weights.call(self.__iteration)
                #self.__iterate_over_matrix(shape_of_weight, self.__update_weight)
                self.__iteration += 1

    def __generate_n_dimensional_tensor(self, vectors):
        """
        Erzeugt einen n-dimensionalen Tensor aus n Vektoren,
        wobei jede Dimension die Multiplikation der Elemente aus den Vektoren darstellt.

        :param vectors: Liste von 1D-Arrays (Vektoren)
        :return: n-dimensionaler Tensor
        """
        # Konvertiere alle Vektoren in NumPy-Arrays
        vectors = [np.array(vec) for vec in vectors]

        # Erstelle die Ausgangsform für Broadcasting
        shape = [len(vec) for vec in vectors]
        grids = np.meshgrid(*vectors, indexing='ij')

        # Multipliziere alle Gitterpunkt-Werte
        result = np.ones(shape)
        for grid in grids:
            result *= grid  # Elementweise Multiplikation

        return result

    def __update_weight(self, ind: list[int]):
        # print(ind)
        # print(self.__weights[*ind])
        self.__weights[*ind] += self.__calculate_delta_value(ind)

    def __calculate_delta_value(self, ind: list[int]) -> float:
        result = 1.0
        # print(f'weightpos: {ind}')
        for i in range(len(ind)):
            # print([i, ind[i]])
            # print(self.__current_activations[i][ind[i]])
            # print(self.__activation_bars[i][ind[i]])
            result *= (self.__current_activations[i][ind[i]] - self.__activation_bars[i][ind[i]])
            # result *= self.__activation_bars[ind[i]]
        result *= self.__lrf_weights.call(self.__iteration)
        # print(f'index: {ind} delta value: {result}')
        return result

    def __iterate_over_matrix_iterative(self, shape, call_back_fn):
        stack = [[]]  # Start mit einer leeren Liste für die Indizes

        while stack:
            indices = stack.pop()

            # Wenn die Länge der Indizes der Dimension der Matrix entspricht, ist das Ziel erreicht
            if len(indices) == len(shape):
                call_back_fn(indices)
            else:
                # Sonst füge weitere Indizes hinzu
                current_dimension = len(indices)
                for i in range(shape[current_dimension]):
                    stack.append(indices + [i])

    def __iterate_over_matrix(self, shape: typing.Tuple[int], call_back_fn: typing.Callable[[list[int]], any],
                              indices=None, current_index=0):
        if indices is None:
            indices = []
        ind = indices.copy()
        if current_index == len(shape):
            call_back_fn(indices)
            return
        new_current_index = current_index + 1
        for i in range(shape[current_index]):
            self.__iterate_over_matrix(shape, call_back_fn, indices=ind + [i], current_index=new_current_index)
