"""
module  : metrics.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains class to get the metrics based on ground truth and predictions

objects:
    - Metrics:
        object which calculates the metrics using actual and predicted data.
        - regression_metrics
            - metrics related to regression problems

"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    mean_absolute_percentage_error,
)


class Metrics:
    """
    A class that calculates various regression metrics for a given dataset.

    Args:
        dataset (pandas.DataFrame): The dataset containing the target and predicted values.
        target (str): The name of the target column in the dataset.
        predicted (str, optional): The name of the predicted column in the dataset. Defaults to "predicted".
    """

    _target = None
    _predicted = None
    _dataset = None
    _mae = None
    _rmse = None
    _r2 = None
    _rmsle = None
    _mape = None

    def __init__(self, dataset, target, predicted="predicted"):
        self._dataset = dataset
        self._target = target
        self._predicted = predicted

    def regression_metrics(self):
        self._mae = mean_absolute_error(
            self._dataset[self._target],
            self._dataset[self._predicted],
        )
        self._rmse = np.sqrt(
            mean_squared_error(
                self._dataset[self._target],
                self._dataset[self._predicted],
            )
        )
        self._r2 = r2_score(
            self._dataset[self._target],
            self._dataset[self._predicted],
        )
        self._rmsle = np.sqrt(
            mean_squared_log_error(
                self._dataset[self._target],
                self._dataset[self._predicted],
            )
        )
        self._mape = mean_absolute_percentage_error(
            self._dataset[self._target],
            self._dataset[self._predicted],
        )

        return {
            "mean_absolute_error": self._mae,
            "root_mean_squared_error": self._rmse,
            "r2": self._r2,
            "root_mean_squared_log_error": self._rmsle,
            "mean_abolute_percentage_error": self._mape,
        }
