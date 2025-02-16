import time
import typing
from typing import Optional, Callable, Literal, Any

import numpy as np
from numpy import ndarray, dtype, float64
from tqdm import tqdm

from . import approximate_local_min_axenie
from .spice_net_hcm import SpiceNetHcm
from .spice_net_som import SpiceNetSom


def _norm(vec: np.array) -> np.array:
    return vec / vec[vec.argmax()]  # TODO: Doc
    # return vec / np.sum(vec)


def _partition_training_data(values: typing.Mapping[str, list[float | np.ndarray[any, np.dtype[np.float64]]]],
                             batch_size: int) -> tuple[
    dict[str, list[list[float | ndarray[Any, dtype[float64]]]]], int]:
    result = {}
    portion_number = 0
    for key, value_set in values.items():
        result[key] = [value_set[i:i + batch_size] for i in range(0, len(value_set), batch_size)]
        portion_number = len(result[key])
    return result, portion_number


class SpiceNet:
    def __init__(self,
                 correlation_matrix: SpiceNetHcm):
        self.__soms = correlation_matrix.get_soms()
        self.__correlation_matrix = correlation_matrix

    def get_correlation_matrix(self):
        return self.__correlation_matrix

    def decode(self,
               values: typing.Mapping[str, float | np.ndarray[any, np.dtype[np.float64]]],
               decoder: Literal["naive", "bert-optimizer"] = "bert-optimizer") -> float:
        existing_keys = values.keys()
        missing_key = None
        som: SpiceNetSom = None
        for som in self.__soms:
            if som.get_key() not in existing_keys:
                if missing_key is None:
                    missing_key = som.get_key()
                    som = som
                else:
                    raise ValueError(f"Found 2 missing keys: {missing_key} and {som.get_key()}")
        if som is None:
            raise ValueError(f"Value for all soms given, no need to decode")

        activation_vectors = {}
        for given_som in self.__soms:
            if given_som.get_key() == missing_key:
                continue
            activation_vectors[given_som.get_key()] = given_som.get_activation_vector(
                values[given_som.get_key()])
        match decoder:

            case "naive":
                should_activation: np.ndarray = self.__correlation_matrix.calculate_should_pattern(missing_key,
                                                                                                   activation_vectors)
                winner_index = should_activation.argmax()
                return som.naive_decode(should_activation[winner_index], winner_index)

            case "bert-optimizer":
                # input is given
                # cost = (normed(som_known(input) * correlation) - normed(som_searched(x)))²
                #      = (normed(should) - normed(som_searched(x)))²

                # cost = som_known(input) - (som_searched(x) * correlation)
                # better:
                # cost = som_known_winning(input) - (som_searched_winning(x) * correlation)
                # tol=(1.0e-6)*(limL + limH)/2.0;
                should_activations = self.__correlation_matrix.calculate_should_pattern(missing_key,
                                                                                        activation_vectors)
                # norm
                normed_should_activations = _norm(should_activations)
                winning_index = normed_should_activations.argmax()

                fn = lambda x: np.linalg.norm(
                    normed_should_activations - _norm(som.get_activation_vector(x))) ** 2  # TODO: Doc quadratic

                start_index = winning_index - 1 if winning_index > 0 else winning_index
                end_index = winning_index + 1 if winning_index < len(normed_should_activations) - 1 else winning_index
                start: float = som.get_as_matrix()[start_index][0]
                end: float = som.get_as_matrix()[end_index][0]
                #print(f'start_index: {start_index}, end_index: {end_index}, winning_index: {winning_index}, start: {start}, end: {end}')
                return approximate_local_min_axenie(start, end, fn, tolerance=abs((1.0e-6) * (start + end) / 2.0))

    def fit(self,
            values: typing.Mapping[str, list[float | np.ndarray[any, np.dtype[np.float64]]]],
            epochs_on_batch: int,
            batch_size: Optional[int] = None,
            after_batch_callback: Optional[Callable] = None,
            print_output: Optional[bool] = False):
        """
        Use this class to train and use a SpiceNet
        :param values:
        :param epochs_on_batch: Define how often a batch is used for training.
        :param batch_size: The values will be split in batches of this size for the training.
        (First the soms will train on ALL the batches data, then the correlation matrix will be trained.)
        :param after_batch_callback: This method will be called after each training with a batch. (Use it for plotting or what ever)
        :param print_output: This will toggle a progressbar implemented with tqdm and stops the computation times.
        :return:
        """
        som_elapsed_time = 0
        cm_elapsed_time = 0
        tmp: float = 0.0

        data_len = -1
        for som in self.__soms:
            if values[som.get_key()] is None:
                raise ValueError('Missing value for ' + som.get_key())
            if data_len != -1 and data_len != len(values[som.get_key()]):
                raise ValueError('The length of the data columns is not allways the same.')
            if data_len == -1:
                data_len = len(values[som.get_key()])

        keys = [som.get_key() for som in self.__soms]

        b_size = data_len if batch_size is None else batch_size
        partitions, partition_count = _partition_training_data(values, b_size)

        iterator = range(partition_count) if print_output is False else tqdm(range(partition_count), colour='green')
        for i in iterator:
            if print_output:
                tmp = time.time()
            for som in self.__soms:
                som.fit(partitions[som.get_key()][i], epochs_on_batch)

            if print_output:
                som_elapsed_time += time.time() - tmp

            if print_output:
                tmp = time.time()

            temp = {key: partitions[key][i] for key in keys}
            self.__correlation_matrix.fit(values=temp,
                                          epochs=epochs_on_batch)
            if print_output:
                cm_elapsed_time += time.time() - tmp

            if after_batch_callback is not None:
                after_batch_callback()

        if print_output:
            print(
                f'Time spend on the Components: \nSom: {som_elapsed_time} s | Convolution Matrix: {cm_elapsed_time} s')
