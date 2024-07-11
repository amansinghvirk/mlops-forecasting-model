"""
module  : deploy.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module deploys the selected model for predictions


Arguments:
    - execution_name: str -> name to group which contains the experiment
    - experiments_id: str -> id of experiment which needs to be deployed as model
    - description: str -> Provide the description for the deployed model
"""

import argparse
import os

from utils.utils import setup, get_proj_dir
from manage.deploy_experiment import copy_files, validate_deploy


def main(execution_name: str, experiment_id: str, description: str) -> str:
    """
    Deploy the model to the specified directory.

    Args:
        execution_name (str): The name of the execution.
        experiment_id (str): The ID of the experiment.
        description (str): The description of the deployment.

    Returns:
        str: The status message indicating whether the model deployment was successful or failed.
    """
    proj_dir = get_proj_dir()

    (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir) = setup(
        proj_dir, execution_name, experiment_id
    )

    src_dir = experiment_path
    dest_dir = deployed_models_dir

    copy_files(src_dir, dest_dir)

    if validate_deploy(src_dir, dest_dir):
        with open(os.path.join(dest_dir, "deployment_desc.txt"), "w") as f:
            f.write(description)
        return "Model deployment successful"
    else:
        return "Model deployment failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution_name", help="name to save the experiment executions"
    )
    parser.add_argument("--experiment_id", help="experiment id to be deployed as model")
    parser.add_argument("--description", help="description about the deployed model")
    args = parser.parse_args()
    execution_name = args.execution_name
    experiments_id = args.experiment_id
    description = args.description
    main(execution_name, experiments_id, description)
