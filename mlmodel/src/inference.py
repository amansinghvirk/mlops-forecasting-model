"""
module  : inference.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains inference functions to be used in execution_api and deploy_api


"""

import os
import json
import yaml
import pandas as pd

from utils.utils import setup, get_db_path, get_proj_dir
from features_pipeline.data_ingestion.prepare_data import Datasets
from features_pipeline.features.features import Features
from inference_pipeline.inference import Inference
from training_pipeline.metrics import Metrics


def get_modeling_data(proj_dir: str, db_path: str, params: dict) -> tuple:
    """
    Retrieves the modeling data for training and validation.

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


def get_prediction_data(
    proj_dir: str, db_path: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves the prediction data for the given project directory, database path, start date, and end date.

    Args:
        proj_dir (str): The project directory.
        db_path (str): The database path.
        start_date (str): The start date for the prediction data.
        end_date (str): The end date for the prediction data.

    Returns:
        pd.DataFrame: The prediction data as a pandas DataFrame.
    """
    datasets = Datasets(proj_dir, db_path)
    features = Features(datasets=datasets)
    features.prepare_inference_data(start_dt=start_date, end_dt=end_date)

    inference_df = features.get_inference_data()

    return inference_df


def model_predictions(df: pd.DataFrame, model_path: str, params: dict) -> pd.DataFrame:
    """
    Generate predictions using a trained model.

    Args:
        df (pd.DataFrame): The input dataset.
        model_path (str): The path to the trained model.
        params (dict): Additional parameters for inference.

    Returns:
        pd.DataFrame: The predictions generated by the model.
    """

    features = params.get("features")
    target = params.get("target")
    inference = Inference(
        dataset=df, modelpath=model_path, features=features, target=target
    )
    predictions = inference.get_predictions()

    return predictions


def get_experiment_params(execution_name: str, experiment_id: str) -> dict:
    """
    Retrieves the experiment parameters from the experiment_params.json file.

    Args:
        execution_name (str): The name of the execution.
        experiment_id (str): The ID of the experiment.

    Returns:
        dict: A dictionary containing the experiment parameters, including name, description,
              model_type, id, target, and features.

    Raises:
        FileNotFoundError: If the experiment_params.json file does not exist.
    """

    predictions = dict()
    proj_dir = get_proj_dir()
    db_path = get_db_path(proj_dir)

    (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir) = setup(
        proj_dir, execution_name, experiment_id
    )
    model_path = os.path.join(model_dir, "model")
    experiment_params_file = os.path.join(experiment_path, "experiment_params.json")
    if os.path.exists(experiment_params_file):
        with open(experiment_params_file, "r") as f:
            params = json.load(f)

        name = params.get("name")
        description = params.get("description")
        model_type = params.get("model_type")
        id_col = params.get("id")
        target = params.get("target")
        features = params.get("features")

        return {
            "name": name,
            "description": description,
            "model_type": model_type,
            "id": id_col,
            "target": target,
            "features": features,
        }
    else:
        raise FileNotFoundError("Experiment params file not found.")


def get_deployed_model_params() -> dict:
    """
    Retrieves the parameters of the deployed model.

    Returns:
        dict: A dictionary containing the parameters of the deployed model, including name, description,
              model_type, id, target, and features.
    """
    proj_dir = get_proj_dir()

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")
    experiment_params_file = os.path.join(deployed_models_dir, "experiment_params.json")
    if os.path.exists(experiment_params_file):
        with open(experiment_params_file, "r") as f:
            params = json.load(f)

        name = params.get("name")
        description = params.get("description")
        model_type = params.get("model_type")
        id_col = params.get("id")
        target = params.get("target")
        features = params.get("features")

        return {
            "name": name,
            "description": description,
            "model_type": model_type,
            "id": id_col,
            "target": target,
            "features": features,
        }
    else:
        print("Experiment not found")


def get_experiment_predictions(execution_name: str, experiment_id: str) -> dict:
    """
    Retrieves the predictions for a given experiment.

    Args:
        execution_name (str): The name of the execution.
        experiment_id (str): The ID of the experiment.

    Returns:
        dict: A dictionary containing the train and valid predictions.
    """
    predictions = dict()
    proj_dir = get_proj_dir()
    db_path = get_db_path(proj_dir)

    (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir) = setup(
        proj_dir, execution_name, experiment_id
    )
    model_path = os.path.join(model_dir, "model")
    experiment_params_file = os.path.join(experiment_path, "experiment_params.json")
    if os.path.exists(experiment_params_file):
        with open(experiment_params_file, "r") as f:
            params = json.load(f)

        id_col = params.get("id")
        target = params.get("target")
        (train_df, valid_df) = get_modeling_data(proj_dir, db_path, params)
        train_predictions = model_predictions(train_df, model_path, params)
        valid_predictions = model_predictions(valid_df, model_path, params)

        cols_to_report = [id_col, target, "predicted"]

        predictions["train"] = train_predictions.loc[:, cols_to_report]
        predictions["valid"] = valid_predictions.loc[:, cols_to_report]

        return predictions
    else:
        print("Experiment not found")


def get_experiment_report(execution_name: str, experiment_id: str) -> dict:
    """
    Retrieves the experiment report for a given execution name and experiment ID.

    Args:
        execution_name (str): The name of the execution.
        experiment_id (str): The ID of the experiment.

    Returns:
        dict: The experiment report in JSON format.

    Raises:
        None
    """
    proj_dir = get_proj_dir()

    (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir) = setup(
        proj_dir, execution_name, experiment_id
    )
    experiment_report = os.path.join(experiment_path, "model_metrics.csv")
    if os.path.exists(experiment_report):
        model_metrics = pd.read_csv(experiment_report)
        model_metrics = model_metrics.to_json(indent=4)
        return model_metrics
    else:
        print("Report not available!")


def get_deployed_model_report() -> dict:
    """
    Retrieves the deployed model report.

    Returns:
        dict: The model report in JSON format.
    """
    proj_dir = get_proj_dir()

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")

    experiment_report = os.path.join(deployed_models_dir, "model_metrics.csv")
    if os.path.exists(experiment_report):
        model_metrics = pd.read_csv(experiment_report)
        model_metrics = model_metrics.to_json(indent=4)
        return model_metrics
    else:
        print("Report not available!")


def get_deployed_predictions() -> dict:
    """
    Retrieves the deployed predictions for the trained model.

    Returns:
        dict: A dictionary containing the predictions for the training and validation datasets.
              The dictionary has two keys: 'train' and 'valid', each containing a DataFrame
              with columns 'id', 'target', and 'predicted'.
    """
    predictions = dict()
    proj_dir = get_proj_dir()
    db_path = get_db_path(proj_dir)

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")
    model_path = os.path.join(deployed_models_dir, "models/model")
    experiment_params_file = os.path.join(deployed_models_dir, "experiment_params.json")
    if os.path.exists(experiment_params_file):
        with open(experiment_params_file, "r") as f:
            params = json.load(f)

        id_col = params.get("id")
        target = params.get("target")
        (train_df, valid_df) = get_modeling_data(proj_dir, db_path, params)
        train_predictions = model_predictions(train_df, model_path, params)
        valid_predictions = model_predictions(valid_df, model_path, params)

        cols_to_report = [id_col, target, "predicted"]

        predictions["train"] = train_predictions.loc[:, cols_to_report]
        predictions["valid"] = valid_predictions.loc[:, cols_to_report]

        return predictions
    else:
        print("Experiment not found")


def predict(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Predicts the values for a given date range using the deployed model.

    Args:
        start_date (str, optional): The start date of the prediction range. Defaults to None.
        end_date (str, optional): The end date of the prediction range. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted values.
    """
    if end_date is None:
        end_date = start_date

    predictions = dict()
    proj_dir = get_proj_dir()
    db_path = get_db_path(proj_dir)

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")
    model_path = os.path.join(deployed_models_dir, "models/model")
    experiment_params_file = os.path.join(deployed_models_dir, "experiment_params.json")
    if os.path.exists(experiment_params_file):
        with open(experiment_params_file, "r") as f:
            params = json.load(f)

        id_col = params.get("id")
        inference_df = get_prediction_data(proj_dir, db_path, start_date, end_date)
        predictions = model_predictions(inference_df, model_path, params)

        cols_to_report = [id_col, "predicted"]

        predictions.loc[:, cols_to_report]

        return predictions
    else:
        print("Experiment not found")


def get_dir_list(root_dir: str) -> list:
    """
    Get a list of directories in the specified root directory.

    Args:
        root_dir (str): The root directory to search for directories.

    Returns:
        list: A list of directories found in the root directory.
    """
    dir_list = []
    for pth in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, pth)):
            dir_list.append(pth)

    return dir_list


def get_model_executions() -> dict:
    """
    Retrieves a dictionary of model executions.

    Returns:
        dict: A dictionary containing the list of model executions.
    """
    executions = dict()
    proj_dir = get_proj_dir()
    # get list of all the models in models directory
    experiments_path = os.path.join(proj_dir, "experiments")
    executions["executions"] = get_dir_list(experiments_path)

    return executions


def get_model_experiments(execution_name: str) -> dict:
    """
    Get a dictionary of model experiments for a given execution name.

    Args:
        execution_name (str): The name of the execution.

    Returns:
        dict: A dictionary containing the model experiments.

    """
    experiments = dict()
    proj_dir = get_proj_dir()
    # get list of all the models in models directory
    experiments_path = os.path.join(proj_dir, "experiments", execution_name)
    experiments["experiments"] = get_dir_list(experiments_path)

    return experiments


def get_deployed_desc() -> str:
    """
    Retrieves the description of the deployed model.

    Returns:
        str: The description of the deployed model.
    """
    proj_dir = get_proj_dir()

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")

    model_desc = os.path.join(deployed_models_dir, "deployment_desc.txt")

    with open(model_desc, "r") as f:
        return f.read()
