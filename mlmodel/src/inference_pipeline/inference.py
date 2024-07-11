"""
module  : inference.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains the logic to use the data pipeline and saved model to get the 
      predictions

objects:
    - Inference:
        object which takes the features dataset and use saved model to get the 
        predictions

"""

from training_pipeline.pycaret import PycaretModel
from training_pipeline.metrics import Metrics


class Inference:
    """
    Class for performing inference using a trained model.

    Args:
        dataset (str): Path to the dataset used for training the model.
        modelpath (str): Path to the trained model.
        features (list): List of feature names used for prediction.
        target (str): Name of the target variable.

    Attributes:
        _dataset (str): Path to the dataset used for training the model.
        _modelpath (str): Path to the trained model.
        _features (list): List of feature names used for prediction.
        _predicted (list): List of predicted values.

    Methods:
        __init__(self, dataset, modelpath, features, target): Initializes the Inference object.
        _predict(self): Performs the prediction using the trained model.
        get_predictions(self): Returns the predicted values.

    """

    _dataset = None
    _modelpath = None
    _features = None
    _predicted = None

    def __init__(self, dataset, modelpath, features, target):
        self._dataset = dataset
        self._modelpath = modelpath
        self._features = features
        self._predict()

    def _predict(self):
        pycaret_model = PycaretModel(self._dataset)
        pycaret_model.prepare_data(features=self._features)
        pycaret_model.load_model(modelpath=self._modelpath)
        pycaret_model.predict()
        self._predicted = pycaret_model.get_predicted()

    def get_predictions(self):
        return self._predicted
