import os
import copy
import torch
from tqdm import tqdm

import torch.nn.functional as F

from .data.datasets import RetrievalTumorDataset
from .models.multimodal import CoattentionModel

import gbheterogeneity.rna_processing.data.datasets as rna_datasets
import gbheterogeneity.image_processing.models.encoder as img_encoder
import gbheterogeneity.rna_processing.models.transformer as rna_encoder

import gbheterogeneity.utils.git.utils as git_utils
import gbheterogeneity.utils.pytorch.initialization as initialization
import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools

from typing import Iterable, Tuple, Dict


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n_correct = (predictions == labels).sum()
    accuracy = n_correct / predictions.shape[0]
    return accuracy


def images_to_device(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device)
    return images


def rna_to_device(
    rna_expression: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    for key, tensor in rna_expression.items():
        rna_expression[key] = tensor.float().to(device)
    return rna_expression


def get_rna_mapping(rna_data: Tuple) -> Dict:
    rna_set = set(rna_data)
    rna_mapping = {}
    for i, rna_category in enumerate(rna_set):
        rna_mapping[rna_category] = i
    return rna_mapping


def map_to_labels(rna_data: Tuple, mapping: Dict, device: torch.device) -> torch.Tensor:
    new_rna_data = []
    for rna_item in rna_data:
        new_rna_data.append(mapping[rna_item])
    new_rna_data = torch.tensor(new_rna_data)
    new_rna_data = new_rna_data.to(device)
    return new_rna_data


def to_label_format(rna_data: Tuple, device: torch.device) -> torch.tensor:
    mapping = get_rna_mapping(rna_data)
    new_rna_data = map_to_labels(rna_data, mapping, device)
    return mapping, new_rna_data


def run_rna_retrieval_on_dataloader(
    model: torch.nn.Module,
    rna_dataloader: Iterable,
    loader: Iterable,
    device: torch.device,
) -> Tuple:
    rna_targets, rna_patients, rna_lineages = next(iter(rna_dataloader))
    mapping_patients, rna_patients = to_label_format(rna_patients, device)
    mapping_lineages, rna_lineages = to_label_format(rna_lineages, device)
    rna_targets = rna_to_device(rna_targets, device)
    representations_rna = model.rna_model.encode(rna_targets)
    projections_rna = F.normalize(
        model.rna_projector(representations_rna[:, 0, :]), dim=-1
    )

    loader_len = len(loader)
    global_patient_accuracy = []
    global_lineage_accuracy = []
    for i, batch in enumerate(tqdm(loader, desc="Computing batch", total=loader_len)):
        # Get image projections
        images, image_patients, image_lineages = batch
        image_patients = map_to_labels(image_patients, mapping_patients, device)
        image_lineages = map_to_labels(image_lineages, mapping_lineages, device)

        images = images_to_device(images, device)
        representations_image, _ = model.img_model(images)
        projections_image = F.normalize(
            model.image_projector(representations_image[:, 0, :]), dim=-1
        )

        # Get predictions
        similarity = projections_image @ projections_rna.t()
        rna_indices = torch.argmax(similarity, dim=1)
        rna_patient_predictions = rna_patients[rna_indices]
        rna_lineages_predictions = rna_lineages[rna_indices]

        # Compute accuracy
        patient_accuracy = compute_accuracy(rna_patient_predictions, image_patients)
        lineage_accuracy = compute_accuracy(rna_lineages_predictions, image_lineages)
        global_patient_accuracy.append(patient_accuracy)
        global_lineage_accuracy.append(lineage_accuracy)

    global_patient_accuracy = torch.tensor(global_patient_accuracy).mean()
    global_lineage_accuracy = torch.tensor(global_lineage_accuracy).mean()
    return global_patient_accuracy, global_lineage_accuracy


def perform_rna_retrieval(
    model: torch.nn.Module,
    rna_dataloader: Iterable,
    train_image_loader: Iterable,
    val_image_loader: Iterable,
    device: torch.device,
) -> None:
    gp_accuracy_val, gl_accuracy_val = run_rna_retrieval_on_dataloader(
        model, rna_dataloader, val_image_loader, device
    )
    mlflow_logging.log_metric(
        key="Accuracy_patient", value=gp_accuracy_val, step=0, prefix="Validation"
    )
    mlflow_logging.log_metric(
        key="Accuracy_lineage", value=gl_accuracy_val, step=0, prefix="Validation"
    )

    gp_accuracy_train, gl_accuracy_train = run_rna_retrieval_on_dataloader(
        model, rna_dataloader, train_image_loader, device
    )
    mlflow_logging.log_metric(
        key="Accuracy_patient", value=gp_accuracy_train, step=0, prefix="Train"
    )
    mlflow_logging.log_metric(
        key="Accuracy_lineage", value=gl_accuracy_train, step=0, prefix="Train"
    )
    return None


def evaluate(config: dict) -> None:
    # Distribution mode
    dist_params = dst_tools.init_distributed_mode()
    device = torch.device(config["device"])

    # Get current commit
    commit = git_utils.get_commit_hash()

    # For reproducibility
    seed = (
        config["manual_seed"] + dist_params["rank"] if dist_params["distributed"] else 0
    )
    initialization.set_deterministic_start(seed)

    # Run Mlflow logger
    logger_params = config["logger_params"]
    mlflow_logging.initialize_mlflow_experiment(logger_params)
    mlflow_logging.log_config(config)
    mlflow_logging.log_commit(commit)
    mlflow_logging.log_environment_name()
    model_dir = mlflow_logging.create_model_directory(logger_params["experiment_name"])
    mlflow_logging.log_model_dir(model_dir)
    run_path = os.path.dirname(model_dir)
    artifacts_dir = os.path.join(run_path, "artifacts")
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    # Get RNA dataset
    rna_dataset_params = config["rna_dataset_params"]
    rna_dataset = rna_datasets.RetrievalRNASeq(
        oncopole_samples_directory=rna_dataset_params["oncopole_samples_directory"],
        tcga_split=rna_dataset_params["tcga_split"],
        gene_cluster_mapping_file=rna_dataset_params["gene_cluster_mapping_file"],
        oncopole_rna_path=rna_dataset_params["oncopole_rna_path"],
    )
    rna_dataloader_params = {"batch_size": len(rna_dataset), "num_workers": 4}
    rna_dataloader = torch.utils.data.DataLoader(
        rna_dataset, **rna_dataloader_params, shuffle=False
    )

    # Get image datasets
    # Get training set
    img_dataset_params = config["img_dataset_params"]
    dataloader_params = config["dataloader_params"]
    image_dataset = RetrievalTumorDataset(
        img_dataset_params=img_dataset_params,
        gene_file_mapping=rna_dataset.gene_file_mapping,
        train=True,
    )
    train_image_loader = torch.utils.data.DataLoader(
        image_dataset, **dataloader_params, shuffle=False
    )

    # Get validation dataset
    val_dataloader_params = config["val_dataloader_params"]
    val_image_dataset = copy.deepcopy(image_dataset)
    val_image_dataset.tumor_dataset.train = False
    val_image_loader = torch.utils.data.DataLoader(
        val_image_dataset, **val_dataloader_params, shuffle=False
    )

    # Get image encoder
    img_model_params = config["img_model_params"]
    img_model = img_encoder.ImageEncoder(img_model_params)

    # Get rna encoder
    rna_model_params = config["rna_model_params"]
    rna_model_params["genes_per_cluster"] = (
        rna_dataset.genes_per_site
        if hasattr(rna_dataset, "genes_per_cluster")
        else None
    )
    rna_model = rna_encoder.AttentionRNA(**rna_model_params)

    # Get multimodal model
    multimodal_model_params = config["multimodal_model_params"]
    model = CoattentionModel(
        **multimodal_model_params, img_model=img_model, rna_model=rna_model
    )

    # Load model
    if config["pretrained_path"]:
        state_dict = torch.load(config["pretrained_path"], map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Loading multimodal model checkpoint from %s" % config["pretrained_path"])

    model = model.to(device)
    model.eval()

    perform_rna_retrieval(
        model, rna_dataloader, train_image_loader, val_image_loader, device
    )
