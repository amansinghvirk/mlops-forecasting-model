import os
from dotenv import dotenv_values


def get_proj_dir():
    proj_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    return proj_dir


def get_db_path(proj_dir):
    config = dotenv_values(os.path.join(proj_dir, ".env"))
    sqlite_db_path = config.get("SQLITEDB_PATH")

    return sqlite_db_path


def setup(proj_dir, execution_name, experiment_uid):
    """
    Sets up the necessary directories for saving experiment results.

    Args:
        proj_dir (str): The project directory.
        execution_name (str): The name of the execution.
        experiment_uid (str): The unique identifier for the experiment.

    Returns:
        tuple: A tuple containing the paths to the experiment directory, model directory, plots directory,
               logs directory, and deployed models directory.
    """
    # Save the experiment results
    if os.path.exists(os.path.join(proj_dir, "experiments")) == False:
        os.mkdir(os.path.join(proj_dir, "experiments"))

    if os.path.exists(os.path.join(proj_dir, "experiments", execution_name)) == False:
        os.mkdir(os.path.join(proj_dir, "experiments", execution_name))

    if (
        os.path.exists(
            os.path.join(proj_dir, "experiments", execution_name, experiment_uid)
        )
        == False
    ):
        os.mkdir(os.path.join(proj_dir, "experiments", execution_name, experiment_uid))

    if (
        os.path.exists(
            os.path.join(
                proj_dir, "experiments", execution_name, experiment_uid, "models"
            )
        )
        == False
    ):
        os.mkdir(
            os.path.join(
                proj_dir, "experiments", execution_name, experiment_uid, "models"
            )
        )

    if (
        os.path.exists(
            os.path.join(
                proj_dir, "experiments", execution_name, experiment_uid, "plots"
            )
        )
        == False
    ):
        os.mkdir(
            os.path.join(
                proj_dir, "experiments", execution_name, experiment_uid, "plots"
            )
        )

    if os.path.exists(os.path.join(proj_dir, "logs")) == False:
        os.mkdir(os.path.join(proj_dir, "logs"))

    if os.path.exists(os.path.join(proj_dir, "logs", execution_name)) == False:
        os.mkdir(os.path.join(proj_dir, "logs", execution_name))

    if (
        os.path.exists(os.path.join(proj_dir, "logs", execution_name, experiment_uid))
        == False
    ):
        os.mkdir(os.path.join(proj_dir, "logs", execution_name, experiment_uid))

    experiment_path = os.path.join(
        proj_dir, "experiments", execution_name, experiment_uid
    )

    if os.path.exists(os.path.join(proj_dir, "deployed_models")) == False:
        os.mkdir(os.path.join(proj_dir, "deployed_models"))

    deployed_models_dir = os.path.join(proj_dir, "deployed_models")

    model_dir = os.path.join(
        proj_dir, "experiments", execution_name, experiment_uid, "models"
    )
    plots_dir = os.path.join(
        proj_dir, "experiments", execution_name, experiment_uid, "plots"
    )
    logs_dir = os.path.join(proj_dir, "logs", execution_name, experiment_uid)

    return (experiment_path, model_dir, plots_dir, logs_dir, deployed_models_dir)


def validate_params(params):
    if (
        (params.get("name") is not None)
        & (params.get("description") is not None)
        & (params.get("model_type") is not None)
        & (params.get("id") is not None)
        & (params.get("target") is not None)
        & (params.get("features") is not None)
        & (params.get("model_params") is not None)
    ):
        return True

    return False
