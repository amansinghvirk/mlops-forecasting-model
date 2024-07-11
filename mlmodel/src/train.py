"""
module  : train.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module to execute the data, features and training pipeline
    - uses the json params files to execute the different experiments

Arguments:
    - execution_name: str -> name to group the different experiments
    - experiments_list: str -> path of json file which contains list of experiments params files

"""

import os
import logging
import json
import yaml
import uuid
import argparse
import pandas as pd

from utils.utils import setup, get_db_path, get_proj_dir, validate_params
from features_pipeline.data_ingestion.prepare_data import Datasets
from features_pipeline.features.features import Features
from training_pipeline.pycaret import PycaretModel
from inference_pipeline.inference import Inference
from training_pipeline.metrics import Metrics
from inference_pipeline.plots import InferencePlots


class NotValidParamsException(Exception):
    "Raised when params in experiment parameters file is invalid"
    pass


class NotValidParamsFileException(Exception):
    "Raised when experiments params file is not available"
    pass


def get_modeling_data(proj_dir: str, db_path: str, params: dict) -> tuple:
    """
    Retrieves and prepares the modeling data for training and validation.

    Args:
        proj_dir (str): The project directory.
        db_path (str): The path to the database.
        params (dict): The parameters for modeling.

    Returns:
        tuple: A tuple containing the training and validation dataframes.
    """
    datasets = Datasets(proj_dir, db_path)
    features = Features(datasets=datasets)
    features.prepare_model_data(
        train_start_dt=params.get("model_params").get("train_start_dt"),
        train_end_dt=params.get("model_params").get("train_end_dt"),
        valid_start_dt=params.get("model_params").get("validation_start_dt"),
        valid_end_dt=params.get("model_params").get("validation_end_dt"),
    )

    train_df, valid_df = features.get_model_data()
    return (train_df, valid_df)


def train_model(
    train_df: pd.DataFrame, logs_dir: str, model_path: str, plots_dir: str, params: dict
) -> dict:
    """
    Trains a machine learning model using the provided training dataset and parameters.

    Args:
        train_df (pd.DataFrame): The training dataset.
        logs_dir (str): The directory to save the logs.
        model_path (str): The path to save the trained model.
        plots_dir (str): The directory to save the plots.
        params (dict): The parameters for training the model.

    Returns:
        dict: A dictionary containing the regression metrics of the trained model on the training dataset.
    """
    pycaret_model = PycaretModel(train_df)
    features = params.get("features")
    target = params.get("target")
    pycaret_model.prepare_data(features=features, target=target)
    pycaret_model.setup_experiment(logfile=os.path.join(logs_dir, "pycaret_logs.csv"))
    pycaret_model.final_model()
    pycaret_model.save_model(modelpath=model_path)

    # predictions on training dataset
    train_inference = Inference(
        dataset=train_df, modelpath=model_path, features=features, target=target
    )
    train_predictions = train_inference.get_predictions()
    train_metrics = Metrics(dataset=train_predictions, target=target)
    train_reg_metrics = train_metrics.regression_metrics()
    InferencePlots(
        dataset=train_predictions,
        id_col=params.get("id"),
        path_to_save=os.path.join(plots_dir, "train_plot.png"),
        target=target,
        predicted="predicted",
    )

    return train_reg_metrics


def validate_model(
    valid_df: pd.DataFrame, model_path: str, plots_dir: str, params: dict
) -> dict:
    """
    Validates the trained model using the provided validation dataset.

    Args:
        valid_df (pd.DataFrame): The validation dataset.
        model_path (str): The path to the trained model.
        plots_dir (str): The directory to save the validation plots.
        params (dict): Additional parameters for validation.

    Returns:
        dict: The regression metrics of the validated model.
    """
    features = params.get("features")
    target = params.get("target")
    valid_inference = Inference(
        dataset=valid_df, modelpath=model_path, features=features, target=target
    )
    valid_predictions = valid_inference.get_predictions()
    valid_metrics = Metrics(dataset=valid_predictions, target=target)
    valid_reg_metrics = valid_metrics.regression_metrics()

    InferencePlots(
        dataset=valid_predictions,
        id_col=params.get("id"),
        path_to_save=os.path.join(plots_dir, "valid_plot.png"),
        target=target,
        predicted="predicted",
    )

    return valid_reg_metrics


def main(execution_name: str, experiments_list: str) -> None:
    """
    Executes the main training process for the ML model.

    Args:
        execution_name (str): The name of the execution.
        experiments_list (str): The path to the file containing the list of experiments.

    Returns:
        None
    """
    proj_dir = get_proj_dir()
    db_path = get_db_path(proj_dir)

    if os.path.exists(os.path.join(proj_dir, "logs")) == False:
        os.mkdir(os.path.join(proj_dir, "logs"))
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(proj_dir, "logs/logs.log"),
        filemode="w",
    )

    with open(os.path.join(proj_dir, experiments_list), "r") as f:
        experiments = yaml.load(f, Loader=yaml.SafeLoader)

    experiment_name = None
    experiment_params_file = None
    experiment_desc = None

    for name, experiment_file in experiments.items():
        experiment_params_file = os.path.join(proj_dir, experiment_file)

        try:
            if not os.path.exists(experiment_params_file):
                raise NotValidParamsFileException("Params file not valid!")
            with open(experiment_params_file, "r") as f:
                params = json.load(f)

        except NotValidParamsFileException as e:
            print(e)
            continue

        try:
            if not validate_params(params):
                raise NotValidParamsException("Not valid params")
        except NotValidParamsException as e:
            print(e)
            continue

        experiment_name = params.get("name")
        print(f"Experiment {experiment_name} executing ...")

        experiment_desc = params.get("description")

        experiment_uid = str(uuid.uuid4())
        (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir) = setup(
            proj_dir, execution_name, experiment_uid
        )
        model_path = os.path.join(model_dir, "model")

        (train_df, valid_df) = get_modeling_data(proj_dir, db_path, params)
        train_metrics = train_model(train_df, logs_dir, model_path, plots_dir, params)
        validate_metrics = validate_model(valid_df, model_path, plots_dir, params)

        # save the results
        model_report = pd.DataFrame(
            {
                "EXPERIMENT_NAME": [experiment_name],
                "EXPERIMENT_DESC": [experiment_desc],
                "TRAIN_MAE": [train_metrics.get("mean_absolute_error")],
                "TRAIN_RMSE": [train_metrics.get("root_mean_squared_error")],
                "TRAIN_R2": [train_metrics.get("r2")],
                "TRAIN_RMSLE": [train_metrics.get("root_mean_squared_log_error")],
                "TRAIN_MAPE": [train_metrics.get("mean_abolute_percentage_error")],
                "VALID_MAE": [validate_metrics.get("mean_absolute_error")],
                "VALID_RMSE": [validate_metrics.get("root_mean_squared_error")],
                "VALID_R2": [validate_metrics.get("r2")],
                "VALID_RMSLE": [validate_metrics.get("root_mean_squared_log_error")],
                "VALID_MAPE": [validate_metrics.get("mean_abolute_percentage_error")],
            }
        )

        model_report.to_csv(
            os.path.join(experiment_path, "model_metrics.csv"), index=False
        )

        # Save execution params
        json_object = json.dumps(params, indent=4)
        with open(os.path.join(experiment_path, "experiment_params.json"), "w") as f:
            f.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution_name", help="name to save the experiment executions"
    )
    parser.add_argument(
        "--experiments_list", help="Yaml file containing list of experiment files"
    )
    args = parser.parse_args()
    execution_name = args.execution_name
    experiments_list = args.experiments_list
    main(execution_name, experiments_list)
