"""
module  : experiment_api.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - FastAPI for the final deployed models

Endpoints:
    - /executions: get -> to get the list of executions
    - /experiments: get -> to get the list of experiments in a execution
        params: 
            - execution_name: str
        end_point
            - /experiemnts/{execution_name}

    - /metrics: get -> to get the list of metrics in a experiment
        params: 
            - execution_name: str
            - execution_id: str
        end_point:
            /metrics/{execution_name}/{experiment_id}

    - /params: get -> to get the list of params in a experiment
        params: 
            - execution_name: str
            - execution_id: str
        end_point:
            /params/{execution_name}/{experiment_id}

    - /predictions: get -> to get the predictions on trian and validate dataset
        params:
            - execution_name: str
            - execution_id: str
        end_point:
            /predictions/{execution_name}/{experiment_id}

    - /deploy: post -> post request to deploy the model as final model for predictions
        data:
            - execution_name: str
            - execution_id: str
            - descriptions: str
        end_point:
            /deploy

Run:
    uvicorn experiment_api:app --port <port-number> --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from inference import (
    get_model_executions,
    get_model_experiments,
    get_experiment_report,
    get_experiment_params,
    get_experiment_predictions,
)
import deploy

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello MLOps world! Model - Stores A"}


@app.get("/executions")
async def executions():
    executions = get_model_executions()
    return executions


@app.get("/experiments/{execution_name}")
async def experiments(execution_name: str):
    experiments = get_model_experiments(execution_name)
    return experiments


@app.get("/metrics/{execution_name}/{experiment_id}")
async def experiment_metrics(execution_name: str, experiment_id: str):
    metrics = get_experiment_report(execution_name, experiment_id)
    return metrics


@app.get("/params/{execution_name}/{experiment_id}")
async def experiment_params(execution_name: str, experiment_id: str):
    params = get_experiment_params(execution_name, experiment_id)
    return params


@app.get("/predictions/{execution_name}/{experiment_id}")
async def experiment_predictions(execution_name: str, experiment_id: str):
    params = get_experiment_predictions(execution_name, experiment_id)
    params["train"] = params["train"].to_json()
    params["valid"] = params["valid"].to_json()
    return params


class Experiment(BaseModel):
    execution_name: str
    experiment_id: str
    description: str


@app.post("/deploy")
async def deploy_experiment(data: Experiment):
    """
    Deploy an experiment.

    Args:
        data (Experiment): The experiment data.

    Returns:
        dict: A dictionary containing the response message.
    """
    data = data.dict()
    execution_name = data["execution_name"]
    experiment_id = data["experiment_id"]
    description = data["description"]
    response = deploy.main(execution_name, experiment_id, description)

    return {"message": response}
