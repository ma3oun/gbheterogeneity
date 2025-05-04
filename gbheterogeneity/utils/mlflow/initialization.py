import mlflow

from typing import Dict


def initialize_mlflow_experiment(config: Dict):
    mlflow.set_tracking_uri(config["experiment_dir"])
    mlflow.set_experiment(config["experiment_name"])
    mlflow.log_params(config)
