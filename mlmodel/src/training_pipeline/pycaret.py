"""
module  : pycaret.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains model training logic using PyCaret python module

objects:
    - PycaretModel:
        object which contains methods to train and save model

"""

import pandas as pd
from pycaret.regression import RegressionExperiment
from pycaret.regression import load_model, predict_model


class PycaretModel:
    """
    A class representing a PyCaret model.

    Attributes:
        _dataset (pandas.DataFrame): The dataset used for training and prediction.
        _features (list): The list of feature columns.
        _target (str): The target column.
        _exp (RegressionExperiment): The PyCaret regression experiment.
        _best_model: The best model selected by PyCaret.
        _final_model: The final trained model.

    Methods:
        __init__(self, dataset): Initializes the PycaretModel instance.
        prepare_data(self, features, target=None): Sets the feature and target columns.
        setup_experiment(self, logfile): Sets up the PyCaret regression experiment.
        final_model(self): Compares baseline models and selects the best model.
        save_model(self, modelpath): Saves the final model to a file.
        load_model(self, modelpath): Loads a saved model from a file.
        predict(self): Generates predictions using the final model.
        get_predicted(self): Returns the dataset with predicted values.
    """

    _dataset = None
    _features = None
    _target = None
    _exp = None
    _best_model = None
    _final_model = None

    def __init__(self, dataset):
        self._dataset = dataset

    def prepare_data(self, features, target=None):
        self._features = features
        self._target = target

    def setup_experiment(self, logfile):
        self._exp = RegressionExperiment()
        # init setup on exp
        self._exp.setup(
            data=self._dataset.loc[:, self._features + [self._target]],
            target=self._target,
            session_id=123,
            system_log=logfile,
        )

    def final_model(self):
        # compare baseline models
        self._best_model = self._exp.compare_models()
        self._final_model = self._best_model

    def save_model(self, modelpath):
        self._exp.save_model(self._final_model, model_name=modelpath)

    def load_model(self, modelpath):
        self._final_model = load_model(modelpath)

    def predict(self):
        predictions = predict_model(
            self._final_model,
            data=self._dataset.loc[:, self._features],
        )
        self._dataset = pd.concat(
            [self._dataset, predictions.loc[:, ["prediction_label"]]], axis=1
        )
        self._dataset.rename(columns={"prediction_label": "predicted"}, inplace=True)

    def get_predicted(self):
        return self._dataset
