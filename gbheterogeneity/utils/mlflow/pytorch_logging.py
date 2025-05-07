import os
import torch
import mlflow

import torchvision.utils as utils

from typing import Dict, Any, List


def initialize_mlflow_experiment(config: Dict) -> None:
    mlflow.set_tracking_uri(uri=config["experiment_dir"])
    mlflow.set_experiment(experiment_name=config["experiment_name"])


def log_param(key: str, value: Any, prefix="") -> None:
    if prefix:
        key = prefix + "/" + key
    if value is not None:
        print("Saving param: {} = {}".format(key, value))
        mlflow.log_param(key=key, value=value)


def log_params(params: Dict[str, Any], prefix="") -> None:
    for key, value in params.items():
        if isinstance(value, (int, float, str, list)):
            log_param(key=key, value=value, prefix=prefix)
        elif isinstance(value, dict):
            log_params(params=value, prefix=prefix + "/" + key)
        else:
            print("Cannot log param: {}".format(key))


def log_config(config: Dict) -> None:
    print("Logging config params")
    for key, value in config.items():
        if isinstance(value, (int, float, str)):
            log_param(key=key, value=value)
        elif isinstance(value, dict):
            log_params(params=value, prefix=key)
        else:
            print("Cannot log param: {}".format(key))


def log_environment_name() -> None:
    print("Saving environment name")
    env_name = os.environ["CONDA_DEFAULT_ENV"]
    mlflow.log_param(key="env_name", value=env_name)


def log_filename(config_file: str) -> None:
    print("Saving config file path: {}".format(config_file))
    mlflow.log_param(key="config_path", value=config_file)


def log_model_dir(model_dir: str) -> None:
    print("Saving model directory path: {}".format(model_dir))
    mlflow.log_param(key="model_dir", value=model_dir)


def log_commit(commit: str) -> None:
    print("Saving commit hash: {}".format(commit))
    mlflow.log_param(key="commit_hash", value=commit)


def log_metric(key: str, value: torch.Tensor, step: int, prefix: str = "") -> None:
    if prefix:
        key = prefix + "/" + key
    if value is not None:
        mlflow.log_metric(key=key, value=value.item(), step=step)


def log_metrics(metrics: Dict[str, torch.Tensor], step: int, prefix="") -> None:
    new_metrics = {}
    for key, value in metrics.items():
        if prefix:
            key = prefix + "/" + key
        if value is not None:
            new_metrics[key] = value.item()

    mlflow.log_metrics(metrics=new_metrics, step=step)


def log_image(image: torch.Tensor, image_file: str, prefix: str = "") -> None:
    if prefix:
        image_file = prefix + "_" + image_file
    np_image = image.cpu().numpy().transpose((1, 2, 0))
    mlflow.log_image(image=np_image, artifact_file=image_file)


def log_image_batch(
    image: torch.Tensor, image_file: str, n_images=8, n_rows=8, prefix=""
) -> None:
    if prefix:
        image_file = prefix + "_" + image_file
    image = utils.make_grid(image[:n_images, :, :, :], nrow=n_rows)
    log_image(image=image, image_file=image_file, prefix=prefix)


def get_experiment(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment


def get_experiment_path(experiment_name: str) -> str:
    experiment = get_experiment(experiment_name=experiment_name)
    return experiment.artifact_location


def get_current_run() -> str:
    run = mlflow.active_run()
    return run.info.run_id


def get_run_path(experiment_name: str) -> str:
    experiment_path = get_experiment_path(experiment_name=experiment_name)
    if experiment_path.startswith("file://"):
        experiment_path = experiment_path[7:]
    run_id = get_current_run()
    run_path = os.path.join(experiment_path, run_id)

    return run_path


def create_model_directory(experiment_name: str) -> str:
    run_path = get_run_path(experiment_name=experiment_name)
    model_dir = os.path.join(run_path, "saved_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir


def log_metrics_list(metrics_list: List[Dict], step: int, prefix="") -> None:
    length = len(metrics_list)
    if length:
        keys = list(metrics_list[0].keys())
        for key in keys:
            list_value = [
                metric[key] for metric in metrics_list if metric[key] is not None
            ]
            if len(list_value):
                avg_value = torch.stack(list_value).mean()
                log_metric(key, avg_value, step, prefix)
