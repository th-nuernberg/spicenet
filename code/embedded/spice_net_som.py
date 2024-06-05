import math
from typing import Optional, Self

import numpy as np


def update_neighbor(current_learning_rate, dampening, distances, node):
    node.preferred_value += current_learning_rate * distances[node] * dampening
    node.tuning_curve_width += dampening


def spice_net_som_activation():
    return (1 / (math.sqrt(2 * math.pi) * 3)) * np.exp(-np.square(np.array(range(-5, 5))))


class SpiceNetSom:
    """
    This self organizing map implementation is specifically for SpiceNet. It creates a 1D som.
    For more information read the theory papers included.
    """

    def __init__(self, nodes: int, value_range_start: float, value_range_end: float):
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
        self.__matrix = np.array([[node.preferred_value for node in self.__nodes],
                                  [node.tuning_curve_width for node in self.__nodes]]).transpose()

    def print_nodes(self):
        for node in self.__nodes:
            print(node)

    def fit(self, values: list[float]):
        for i in range(len(values)):
            current_learning_rate = 1 - i / len(values)

            closest_node, distances = self.__get_closest_node_and_distances(values[i])
            closest_node.preferred_value += current_learning_rate * distances[closest_node]
            closest_node.tuning_curve_width += 1

            node = closest_node.next
            dampening = 1
            while node is not None:
                dampening /= 2
                update_neighbor(current_learning_rate, dampening, distances, node)
                node = node.next

            node = closest_node.previous
            dampening = 1
            while node is not None:
                dampening /= 2
                update_neighbor(current_learning_rate, dampening, distances, node)
                node = node.previous
        self.__matrix = np.array([[node.preferred_value for node in self.__nodes],
                                  [node.tuning_curve_width for node in self.__nodes]]).transpose()

    def get_as_matrix(self) -> np.ndarray:
        """
        Returns all nodes like a table with column
        :return:
        """
        return self.__matrix

    def __get_closest_node_and_distances(self, value: float):
        result = self.__nodes[0]
        min_distance = abs(value - self.__nodes[0].preferred_value)
        distances = {self.__nodes[0]: value - self.__nodes[0].preferred_value}

        for i in range(1, len(self.__nodes)):
            new_distance = value - self.__nodes[i].preferred_value
            distances[self.__nodes[i]] = value - self.__nodes[i].preferred_value
            new_distance = abs(new_distance)
            if new_distance < min_distance:
                min_distance = new_distance
                result = self.__nodes[i]
        return result, distances

    def __get_activation_and_argmax(self, value: float):
        result = self.__nodes[0]
        min_distance = abs(value - self.__nodes[0].preferred_value)
        distances = {self.__nodes[0]: value - self.__nodes[0].preferred_value}

        for i in range(1, len(self.__nodes)):
            new_distance = value - self.__nodes[i].preferred_value
            distances[self.__nodes[i]] = value - self.__nodes[i].preferred_value
            new_distance = abs(new_distance)
            if new_distance < min_distance:
                min_distance = new_distance
                result = self.__nodes[i]
        return result, distances

    class __SomNode:
        """
        This object represents a som node. Since this som is 1D the node only knows th previous and next node.
        """

        def __init__(self,
                     preferred_value: float,
                     tuning_curve_width: float = 1):
            """
            Creates a som node.
            :param preferred_value: The "expected value" of the Normal distribution representing the activation function.
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
