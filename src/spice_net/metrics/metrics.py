import math

import numpy as np


def __validate_input(predicted: np.ndarray, actual: np.ndarray):
    if predicted.shape == (0,) or actual.shape == (0,):
        raise ValueError('arrays must contain values')
    if len(predicted.shape) > 1 or len(actual.shape) > 1:
        raise ValueError('only 1D arrays are supported')
    if predicted.size != actual.size:
        raise ValueError('predicted array and actual array size do not match')


def bias_error(predicted: float, actual: float) -> float:
    return predicted - actual


def mean_bias(predicted: np.ndarray, actual: np.ndarray):
    __validate_input(predicted, actual)
    return np.mean(predicted - actual)


def median_bias(predicted: np.ndarray, actual: np.ndarray):
    __validate_input(predicted, actual)
    return np.median(predicted - actual)


def standard_deviation_bias(predicted: np.ndarray, actual: np.ndarray):
    __validate_input(predicted, actual)
    return np.std(predicted - actual)


def mean_absolute_gross_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.mean(np.abs(predicted - actual))


def mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.mean(np.square(predicted - actual))


def root_mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.sqrt(mean_squared_error(predicted, actual))


def centered_mean_square_difference(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.mean(np.square((predicted - predicted.mean()) - (actual - actual.mean())))


def centered_root_mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.sqrt(centered_mean_square_difference(predicted, actual))


def mean_normalized_bias(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.mean(predicted / actual) - 1


def mean_normalized_gross_error(predicted: np.ndarray, actual: np.ndarray):
    __validate_input(predicted, actual)
    return np.mean(np.abs(predicted - actual) / actual)


def normalized_mean_bias(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return np.mean(predicted) / np.mean(actual) - 1


def normalized_mean_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return mean_absolute_gross_error(predicted, actual) / np.mean(actual)


def fractional_bias(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return 2 / predicted.size * np.sum((predicted - actual) / (predicted + actual))


def fractional_gross_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return 2 / predicted.size * np.sum(np.abs(predicted - actual) / (predicted + actual))


def theils_ui(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return root_mean_squared_error(predicted, actual) / (
            math.sqrt(np.mean(predicted ** 2)) + math.sqrt(np.mean(actual ** 2)))


def index_of_agreement(predicted: np.ndarray, actual: np.ndarray) -> float:
    __validate_input(predicted, actual)
    return 1 - ((predicted.size * mean_squared_error(predicted, actual)) / (
        np.sum(
            np.square(np.abs(predicted - np.mean(actual)) + np.abs(actual - np.mean(actual)))
        )
    ))


def metrics_dict(predicted: np.ndarray, actual: np.ndarray) -> dict[str, any]:
    __validate_input(predicted, actual)
    return {
        'Mean Bias': mean_bias(predicted, actual),
        'Median Bias': median_bias(predicted, actual),
        'Standard Deviation of Bias': standard_deviation_bias(predicted, actual),
        'Mean Absolute Gross Error': mean_absolute_gross_error(predicted, actual),
        'Mean Squared Error': mean_squared_error(predicted, actual),
        'Root Mean Squared Error': root_mean_squared_error(predicted, actual),
        'Centered Mean Square Difference': centered_mean_square_difference(predicted, actual),
        'Centered Root Mean Square Error': centered_root_mean_squared_error(predicted, actual),
        'Mean Normalized Bias': mean_normalized_bias(predicted, actual),
        'Mean Normalized Gross Error': mean_normalized_gross_error(predicted, actual),
        'Normalized Mean Bias': normalized_mean_bias(predicted, actual),
        'Normalized Mean Error': normalized_mean_error(predicted, actual),
        'Fractional Bias': fractional_bias(predicted, actual),
        'Fractional Gross Error': fractional_gross_error(predicted, actual),
        'Theil’s UI': theils_ui(predicted, actual),
        'Index of agreement': index_of_agreement(predicted, actual),
    }
