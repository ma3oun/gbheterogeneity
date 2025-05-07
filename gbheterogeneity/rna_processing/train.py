import torch
import importlib

import gbheterogeneity.utils.git.utils as git_utils
import gbheterogeneity.utils.pytorch.initialization as initialization
import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools


def train(config: dict) -> None:
    # Distribution mode
    dist_params = dst_tools.init_distributed_mode()
    device = torch.device(config["trainer_params"]["device"])

    # Get current commit
    commit = git_utils.get_commit_hash()

    # For reproducibility
    seed = (
        config["manual_seed"] + dist_params["rank"] if dist_params["distributed"] else 0
    )
    initialization.set_deterministic_start(seed)

    # Run Mlflow logger
    if dst_tools.is_main_process():
        logger_params = config["logger_params"]
        mlflow_logging.initialize_mlflow_experiment(logger_params)
        mlflow_logging.log_config(config)
        mlflow_logging.log_commit(commit)
        mlflow_logging.log_environment_name()
        model_dir = mlflow_logging.create_model_directory(
            logger_params["experiment_name"]
        )
        mlflow_logging.log_model_dir(model_dir)
    else:
        model_dir = ""

    # Create datasets
    num_gpus = dst_tools.get_world_size()
    global_rank = dst_tools.get_rank()

    dataset_name = config["dataset_name"]
    dataset_params = config["dataset_params"]
    dataloader_params = config["dataloader_params"]

    # Dynamically import the dataset module and get the class by name
    data_module = importlib.import_module("gbheterogeneity.rna_processing.data")
    train_dataset_getter = getattr(data_module, dataset_name)

    # Initialize the dataset and create the train_loader
    train_loader = train_dataset_getter(
        dataset_params, dataloader_params, num_gpus, global_rank, train=True
    )

    val_dataset_name = config["val_dataset_name"]
    val_dataset_params = config["val_dataset_params"]
    val_dataloader_params = config["val_dataloader_params"]
    val_dataset_getter = getattr(data_module, val_dataset_name)
    val_loader = val_dataset_getter(
        val_dataset_params, val_dataloader_params, num_gpus, global_rank, train=False
    )

    # Get model
    model_name = config["model_name"]
    model_params = config["model_params"]
    model_params["genes_per_cluster"] = (
        train_loader.genes_per_cluster
        if hasattr(train_loader, "genes_per_cluster")
        else None
    )
    model_module = importlib.import_module("gbheterogeneity.rna_processing.models")
    model_registry = getattr(model_module, model_name)
    model = model_registry(**model_params)
    model = model.to(device)

    # Get optimizer
    optimizer_name = config["optimizer_name"]
    optimizer_params = config["optimizer_params"]
    optimizer_module = importlib.import_module("gbheterogeneity.utils.pytorch")
    optim = getattr(optimizer_module, optimizer_name)
    optimizer = optim(model, **optimizer_params)

    # Get scheduler
    scheduler_name = config["scheduler_name"]
    scheduler_params = config["scheduler_params"]
    scheduler_module = importlib.import_module("gbheterogeneity.utils.pytorch")
    scheduler_getter = getattr(scheduler_module, scheduler_name)
    scheduler = scheduler_getter(optimizer, **scheduler_params)

    # Get trainer
    trainer_name = config["trainer_name"]
    logging_params = config["logging_params"]
    trainer_params = config["trainer_params"]
    trainer_module = importlib.import_module("gbheterogeneity.rna_processing.trainers")
    trainer_getter = getattr(trainer_module, trainer_name)
    trainer = trainer_getter(
        logging_params,
        trainer_params,
        save_directory=model_dir,
        dist_params=dist_params,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print("======= Training RNA data =======")
    trainer.fit(model, train_loader, val_loader)
