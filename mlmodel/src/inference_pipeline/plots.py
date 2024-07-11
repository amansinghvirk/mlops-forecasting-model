"""
module  : plots.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains the logic to create plots for analyzing the model performance

objects:
    - InferencePlots:
        object which contains the methods to create plots

"""

import matplotlib.pyplot as plt


class InferencePlots:
    """
    Class for generating inference plots.

    Attributes:
        _id (str): The column name for the ID.
        _dataset (pandas.DataFrame): The dataset used for plotting.
        _path_to_save (str): The path to save the generated plot.
        _target (str): The column name for the target variable.
        _predicted (str): The column name for the predicted variable.

    Methods:
        __init__(self, dataset, id_col, path_to_save, target=None, predicted=None):
            Initializes the InferencePlots object.
        _timeseries_plot(self):
            Generates a timeseries plot based on the provided dataset and column names.
    """

    _id = None
    _dataset = None
    _path_to_save = None
    _target = None
    _predicted = None

    def __init__(self, dataset, id_col, path_to_save, target=None, predicted=None):
        self._dataset = dataset
        self._id = id_col
        self._path_to_save = path_to_save
        self._target = target
        self._predicted = predicted
        self._timeseries_plot()

    def _timeseries_plot(self):
        """
        Generates a timeseries plot based on the provided dataset and column names.

        If both target and predicted column names are provided, the plot will include both.
        If only the predicted column name is provided, the plot will only include the predicted values.
        If only the target column name is provided, the plot will only include the target values.

        The plot will be saved to the specified path.
        """
        cols = None
        if self._target and self._predicted:
            cols = [self._target, self._predicted]
        elif self._predicted:
            cols = [self._predicted]
        elif self._target:
            cols = [self._target]

        if len(cols) > 0:
            # Adjusting the figure size
            fig = plt.subplots(figsize=(16, 5))

            df = self._dataset
            # Adding a plot title and customizing its font size
            plt.title("Model Predictions", fontsize=20)
            for col in ["sales", "predicted"]:
                plt.plot(df[self._id], df[col], label=col)
                plt.xlim(df[self._id].min(), df[self._id].max())
            plt.legend()
            plt.savefig(self._path_to_save)
