import copy
import torch
import importlib

from .data.datasets import RNATumorDataset
from .models.multimodal import CoattentionModel
from .trainer import Trainer

import gbheterogeneity.image_processing.models.encoder as img_encoder
import gbheterogeneity.rna_processing.models.transformer as rna_encoder
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
        config["trainer_params"]["manual_seed"] + dist_params["rank"]
        if dist_params["distributed"]
        else 0
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

    rna_dataset_params = config["rna_dataset_params"]
    img_dataset_params = config["img_dataset_params"]
    dataloader_params = config["dataloader_params"]
    dataset = RNATumorDataset(rna_dataset_params, img_dataset_params, True)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_gpus, rank=global_rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, **dataloader_params, shuffle=False, pin_memory=True, sampler=sampler
    )

    # Get validation dataset
    val_dataloader_params = config["val_dataloader_params"]
    val_dataset = copy.deepcopy(dataset)
    val_dataset.tumor_dataset.train = False
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **val_dataloader_params, shuffle=False
    )

    # Get image encoder
    img_model_params = config["img_model_params"]
    img_model = img_encoder.ImageEncoder(img_model_params)

    # Load model
    if img_model_params["img_pretrained_path"]:
        state_dict = torch.load(
            img_model_params["img_pretrained_path"], map_location="cpu"
        )
        img_model.load_state_dict(state_dict, strict=False)
        print(
            "Loading TumorImage checkpoint from %s"
            % img_model_params["img_pretrained_path"]
        )

    # Get rna encoder
    rna_model_params = config["rna_model_params"]
    rna_model_params["genes_per_cluster"] = (
        dataset.genes_per_cluster if hasattr(dataset, "genes_per_cluster") else None
    )
    rna_model = rna_encoder.AttentionRNA(**rna_model_params)

    # Load model
    if rna_model_params["rna_pretrained_path"]:
        state_dict = torch.load(
            rna_model_params["rna_pretrained_path"], map_location="cpu"
        )
        rna_model.load_state_dict(state_dict, strict=False)
        print(
            "Loading RNA checkpoint from %s" % rna_model_params["rna_pretrained_path"]
        )

    # Get multimodal model
    multimodal_model_params = config["multimodal_model_params"]
    model = CoattentionModel(
        **multimodal_model_params, img_model=img_model, rna_model=rna_model
    )
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

    print("======= Training on RNA and tumor images =======")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
