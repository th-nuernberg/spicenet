import math
from typing import Optional, Self

import numpy as np


class SpiceNetSom:
    """
    This self organizing map implementation is specifically for SpiceNet. It creates a 1D som.
    For more information read the theory papers included.
    """

    def __init__(self, nodes: int, value_range_start: float, value_range_end: float):
        """
        Creates a SpiceNetSom with nodes equally distributed across the specified value range.
        :param nodes: The number of nodes that should be created.
        :param value_range_start: Start of the value range.
        :param value_range_end: End of the value range.
        """
        distance = value_range_end - value_range_start
        step_size = distance / nodes

        self.__nodes = []
        pos = value_range_start + step_size / 2.0
        previous = None

        for i in range(nodes):
            new_node = SpiceNetSom.__SomNode(pos)
            new_node.previous = previous
            if len(self.__nodes) != 0:
                self.__nodes[-1].next = new_node
            self.__nodes.append(new_node)
            previous = new_node
            pos += step_size

    def print_nodes(self):
        for node in self.__nodes:
            print(node)

    def fit(self, values: list[float]):
        sigma_step_size = (len(values) / 2 - 1) / len(values)
        current_sigma = len(values) / 2
        for i in range(len(values)):
            current_learning_rate = 1 - i / len(values)
            current_sigma = len(values) / 2

            winning_node_index, _ = self.__argmax_node_activation(values[i])

            for j in range(len(self.__nodes)):
                self.__nodes[j].update(values[i], current_learning_rate, 1, j - winning_node_index)

            current_sigma -= sigma_step_size

    def get_as_matrix(self) -> np.ndarray:
        """
        Returns all nodes like a table with column 0 representing the preferred value
        and column 1 representing the tuning curve width.
        :return: A numpy matrix.
        """
        return np.array([[node.preferred_value for node in self.__nodes],
                         [node.tuning_curve_width for node in self.__nodes]]).transpose()

    def calculate_activation_values(self, values: list[float]):
        activation_values = np.array([node.activation_for_values(values) for node in self.__nodes])
        return np.concatenate((
            np.array(
                [[node.preferred_value for node in self.__nodes],
                 [node.tuning_curve_width for node in self.__nodes]]).transpose(),
            activation_values),
            axis=1)

    def __argmax_node_activation(self, value: float):
        """
        Calculates the node with the highest activation value.
        :param value: The value for wich the activation values should be calculated.
        :return: The index of the winning node
        and a dictionary containing all nodes with the calculated activation values.
        """
        winning_node_index: int = 0
        max_activation = self.__nodes[0].activation_for_value(value)
        activation_dict = {self.__nodes[0]: max_activation}

        for i in range(1, len(self.__nodes)):
            new_activation = self.__nodes[i].activation_for_value(value)
            # add new value to a dictionary so the activations only have to be calculated a single time
            activation_dict[self.__nodes[i]] = new_activation

            if new_activation > max_activation:
                max_activation = new_activation
                winning_node_index = i
        return winning_node_index, activation_dict

    class __SomNode:
        """
        This object represents a som node. Since this som is 1D the node only knows th previous and next node.
        """

        def __init__(self,
                     preferred_value: float,
                     tuning_curve_width: float = 0.001):
            """
            Creates a som node.
            :param preferred_value: The "expected value" of the Normal distribution
            representing the activation function.
            :param tuning_curve_width: The width of the tuning curve (the Normal distribution).
            """
            self.preferred_value = preferred_value
            """ The "expected value" of the Normal distribution representing the activation function. """
            self.tuning_curve_width = tuning_curve_width
            """ The width of the tuning curve (the Normal distribution). """
            self.previous: Optional[Self] = None
            """ The previous node of the som."""
            self.next: Optional[Self] = None
            """ The next node of the som."""

        def __str__(self):
            previous_str = 'None' if self.previous is None else self.previous.preferred_value
            next_str = 'None' if self.next is None else self.next.preferred_value
            return (f'SomeNode: preferred_value={self.preferred_value}, tuning_curve_width={self.tuning_curve_width}, '
                    f'previous={previous_str}, next={next_str}')

        def activation_for_value(self, value: float):
            """
            Returns the activation value of the node.
            :param value: The value for wich the activation has to be calculated.
            :return: The "height" of the tuning curve at the position of the value.
            """
            return (
                    (1.0 / (math.sqrt(2.0 * math.pi) * self.tuning_curve_width))
                    *
                    math.exp(
                        (-(value - self.preferred_value) ** 2) /
                        (2.0 * self.tuning_curve_width ** 2))
            )

        def activation_for_values(self, values: list[float]):
            """
            Returns the activation values for a list of values.
            :param values: The values for wich the activation has to be calculated.
            :return: An array of activation values, ordered like the input list.
            """
            return np.array([self.activation_for_value(value) for value in values])

        def update(self, value: float, learn_rate: float, sigma: float, distance_to_winner: int):
            """
            Updates the weights of the nodes.
            :param value: The value in wich "direction" the node should move.
            :param learn_rate: The learn rate. (This value should be depending on the iteration.)
            :param sigma: The sigma of the interaction kernel. (This value should be depending on the iteration.)
            :param distance_to_winner: The distance between this node and the winning node, in the SOM.
            Here are not the weights / postions in the value range relevant.
            (N1 weight: 31.9) --- (N2 weight: 32) --- (N3 weight: 32.4) --- (N4 weight: 33)
            The Distance of N4 to N1 is 3
            :return:
            """
            interaction_kernel_value = math.exp((-abs(distance_to_winner) ** 2) / (2 * sigma ** 2))
            self.preferred_value += learn_rate * interaction_kernel_value * (value - self.preferred_value)
            self.tuning_curve_width += learn_rate * interaction_kernel_value * (
                    (value - self.preferred_value) ** 2 - self.tuning_curve_width ** 2
            )
