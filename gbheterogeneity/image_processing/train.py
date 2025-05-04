import torch
import importlib
from torchvision.transforms import ToTensor, Compose, Normalize

import gbheterogeneity.utils.git.utils as git_utils
import gbheterogeneity.utils.pytorch.initialization as initialization
import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools

from .data.datasets import TumorDataset
from .models.encoder import ImageEncoder
from .trainer import Trainer

def train(config:dict) -> None:
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

    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    datasetTransforms = Compose([ToTensor(), normalize])
    dataset_params = config["dataset_params"]
    dataloader_params = config["dataloader_params"]

    dataset = TumorDataset(
        dataset_params["path"],
        dataset_params["patchParams"],
        True,
        datasetTransforms,
        dataset_params["patients"],
        dataset_params["lineage"],
    )

    if dist_params["distributed"]:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=global_rank, shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, **dataloader_params, shuffle=True
        )

    val_dataloader_params = config["val_dataloader_params"]
    val_dataset = TumorDataset(
        dataset_params["path"],
        dataset_params["patchParams"],
        False,
        datasetTransforms,
        dataset_params["patients"],
        dataset_params["lineage"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **val_dataloader_params, shuffle=False, drop_last=True
    )

    # Get model
    model_params = config["model_params"]
    model = ImageEncoder(model_params)
    model = model.to(device)

    # Get optimizer
    optimizer_name = model_params["optimizer_name"]
    optimizer_params = model_params["optimizer_params"]
    optimizer_module = importlib.import_module("gbheterogeneity.utils.pytorch")
    optim = getattr(optimizer_module, optimizer_name)
    optimizer = optim(model, **optimizer_params)

    # Get scheduler
    scheduler_name = model_params["scheduler_name"]
    scheduler_params = model_params["scheduler_params"]
    scheduler_module = importlib.import_module("gbheterogeneity.utils.pytorch")
    scheduler_getter = getattr(scheduler_module, scheduler_name)
    scheduler = scheduler_getter(optimizer, **scheduler_params)


    # Get trainer
    logging_params = config["logging_params"]
    trainer_params = config["trainer_params"]
    trainer = Trainer(
        **logging_params,
        **trainer_params,
        save_directory=model_dir,
        dist_params=dist_params,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print("======= Training Image data =======")
    trainer.fit(model, train_loader, val_loader)

