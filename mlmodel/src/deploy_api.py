"""
module  : deploy_api.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - FastAPI for the final deployed models

Endpoints:
    - /metrics: get -> to get the metrics of deployed model
    - /params: get -> to get the parameters used while training the model
    - /predictions: get -> to get the predictions on trian and validate dataset
    - /description: get -> to get the description of deployed model
    - /singlepredict: post -> post request to get the predictions of single date
        - parameter 
            - date: str (YYYY-MM-DD)
    - /rangepredict: post -> post request to get the predictions of date range
        - parameter 
            - start_date: str (YYYY-MM-DD)
            - end_date: str (YYYY-MM-DD)

Run:
    uvicorn deploy_api:app --port <port-number> --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from inference import (
    get_deployed_model_report,
    get_deployed_model_params,
    get_deployed_predictions,
    predict,
    get_deployed_desc,
)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello from deployed model!"}


@app.get("/metrics")
async def experiment_metrics():
    metrics = get_deployed_model_report()
    return metrics


@app.get("/params")
async def experiment_params():
    params = get_deployed_model_params()
    return params


@app.get("/predictions")
async def experiment_predictions():
    params = get_deployed_predictions()
    params["train"] = params["train"].to_json()
    params["valid"] = params["valid"].to_json()
    return params


@app.get("/description")
async def deployment_desc():
    desc = get_deployed_desc()
    return {"description": desc}


class SinglePredict(BaseModel):
    date: str


class RangePredict(BaseModel):
    start_date: str
    end_date: str


@app.post("/singlepredict")
async def get_predictions_single(data: SinglePredict):
    """
    Get predictions for a single date.

    Args:
        data (RangePredict): The input data containing the start and end dates.

    Returns:
        dict: A dictionary containing the predictions in JSON format.
    """
    data = data.dict()
    start_date = data["date"]

    predictions = predict(start_date=start_date)
    predictions = predictions.loc[:, ["date", "predicted"]]

    return {"prediction": predictions.to_json()}


@app.post("/rangepredict")
async def get_predictions_range(data: RangePredict):
    """
    Get predictions for a range of dates.

    Args:
        data (RangePredict): The input data containing the start and end dates.

    Returns:
        dict: A dictionary containing the predictions in JSON format.
    """
    data = data.dict()
    start_date = data["start_date"]
    end_date = data["end_date"]

    predictions = predict(start_date=start_date, end_date=end_date)
    predictions = predictions.loc[:, ["date", "predicted"]]

    return {"prediction": predictions.to_json()}
