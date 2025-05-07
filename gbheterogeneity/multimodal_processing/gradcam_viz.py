import os
import copy
import torch
import collections
import mygene
import cv2
from tqdm import tqdm

from torch.nn.functional import cosine_similarity

from PIL import Image
import pyvips

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .data.datasets import RNATumorDataset
from .models.multimodal import CoattentionModel
from gbheterogeneity.rna_processing.data.preprocessing import remove_ensembl_version
import gbheterogeneity.image_processing.models.encoder as img_encoder
import gbheterogeneity.rna_processing.models.transformer as rna_encoder

import gbheterogeneity.utils.git.utils as git_utils
import gbheterogeneity.utils.pytorch.initialization as initialization
import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools

from typing import Iterable, Tuple, List, Dict, Union


def images_to_device(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device)
    return images


def rna_to_device(
    rna_expression: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    for key, tensor in rna_expression.items():
        rna_expression[key] = tensor.float().to(device)
    return rna_expression


def data_to_device(
    images: torch.Tensor, rna: Dict[str, torch.Tensor], device: torch.device
) -> Tuple:
    images = images_to_device(images, device)
    rna = rna_to_device(rna, device)

    return images, rna


def set_require_grad_inputs(
    input_data: Union[Dict[str, torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if isinstance(input_data, torch.Tensor):  # for images
        input_data.requires_grad = True
        return input_data
    else:
        for _, value in input_data.items():  # for RNA
            value.requires_grad = True
    return input_data


def get_rna_gradients(rna_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gradients = {}
    for key, value in rna_data.items():
        gradients[key] = value.grad
    return gradients


def compute_gradcam(
    model: CoattentionModel, batch: Tuple, device: torch.device, gradcam_on_rna: bool
) -> Tuple:
    images, rnas, infos = batch
    batch_size, _, _, _ = images.shape
    images, rnas = data_to_device(images, rnas, device)
    if gradcam_on_rna:
        rnas = set_require_grad_inputs(rnas)
        input_for_gradcam = rnas
    else:
        images = set_require_grad_inputs(images)
        input_for_gradcam = images
    p_image, p_rna, _, vl_outputs, _ = model(images, rnas, register_hook=True)

    positive_vl_outputs = vl_outputs[:batch_size]
    # Get matching scores using cosine similarity
    matching_scores = cosine_similarity(p_image, p_rna, dim=1)

    matching_loss = torch.sum(positive_vl_outputs[:, 1])

    model.zero_grad()
    matching_loss.backward()

    # Get gradients per input
    if gradcam_on_rna:
        data_gradients = get_rna_gradients(input_for_gradcam)
    else:
        data_gradients = input_for_gradcam.grad

    # Get gradients per gene cluster
    with torch.no_grad():
        # Get attention maps and gradients
        grads = (
            model.co_attention.get_attn_gradients()
        )  # batch_size, #heads, #patches, #cluster
        cams = (
            model.co_attention.get_attention_map()
        )  # batch_size, #heads, #patches, #cluster

        # Remove global representations from cams and grads.
        cams = cams[:, :, 0, 1:]
        grads = grads[:, :, 0, 1:].clamp(0)

        # Compute gradcam
        gradcam = cams * grads

        # Average attention heads
        data_maps = []
        for i in range(batch_size):
            gradcam_map = gradcam[i].mean(0).cpu().detach()
            data_maps.append(gradcam_map)
    # return infos as list of dicts
    infos = [dict(zip(infos, t)) for t in zip(*infos.values())]
    return matching_scores, data_maps, data_gradients, infos


def split_gradients_by_sample(grads: Dict, num_samples: int) -> List:
    keys = grads.keys()
    new_grads = []

    for sample_id in range(num_samples):
        data_dict = {}
        for key in keys:
            data_dict[key] = grads[key][sample_id]
        new_grads.append(data_dict)

    return new_grads


def plot_attention_on_gene_clusters(
    rna_map: torch.tensor, name: str, directory: str
) -> None:
    basename = os.path.basename(name)
    save_path = os.path.splitext(basename)[0] + ".png"
    save_path = os.path.join(directory, save_path)
    gene_clusters = [i for i in range(rna_map.shape[0])]
    attention = rna_map.cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.plot(gene_clusters, attention, "ro")
    plt.xlabel("Gene cluster ids")
    plt.ylabel("GradCAM values")
    plt.savefig(save_path)
    plt.close()


def plot_attention_on_genes(genes: torch.tensor, name: str, directory: str) -> None:
    basename = os.path.basename(name)
    save_path = os.path.splitext(basename)[0] + "_genes.png"
    save_path = os.path.join(directory, save_path)
    genes_id = [i for i in range(genes.shape[0])]
    attention = genes.cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.plot(genes_id, attention, "ro")
    plt.xlabel("Gene ids")
    plt.ylabel("GradCAM values")
    plt.savefig(save_path)
    plt.close()


def get_relevant_gene_names(
    gene_names: Dict, gene_cluster_key: str, genes_id: List[int], num_genes: int = 10
) -> List[str]:
    names = gene_names[gene_cluster_key]
    new_names = []
    for i in range(num_genes):
        index = int(genes_id[i])
        new_names.append(names[index])
    return new_names


def get_important_genes_per_sample(
    rna_map: torch.tensor, grad: Dict, name: str, gene_names: Dict, artifacts_dir: str
) -> None:
    cluster = torch.argmax(rna_map)
    cluster_key = str(int(cluster.item()))
    genes = grad[cluster_key]
    genes = genes.abs()
    gene_ids = torch.argsort(genes, descending=True)
    relevant_genes = get_relevant_gene_names(gene_names, cluster_key, gene_ids)
    # for visualization only
    # plot_attention_on_gene_clusters(rna_map, name, artifacts_dir)
    # plot_attention_on_genes(genes, name, artifacts_dir)

    return relevant_genes


def get_important_genes(
    maps: List, grads: Dict, names: List, gene_names: Dict, artifacts_dir: str
) -> None:
    grads = split_gradients_by_sample(grads, len(names))
    important_genes = []
    for rna_map, grad, name in zip(maps, grads, names):
        relevant_genes = get_important_genes_per_sample(
            rna_map, grad, name, gene_names, artifacts_dir
        )
        important_genes = important_genes + relevant_genes
    return important_genes


def process_oncopole_rna_data(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    gene_names: Dict,
    artifacts_dir: str,
    relevant_genes_path: str,
) -> None:
    target_genes = []
    for batch in tqdm(loader, desc="Computing gradcams", total=len(loader)):
        _, maps, grads, infos = compute_gradcam(
            model, batch, device, gradcam_on_rna=True
        )

        names = [info["patchFile"] for info in infos]
        relevant_genes = get_important_genes(
            maps, grads, names, gene_names, artifacts_dir
        )
        target_genes = target_genes + relevant_genes
    relevance_occurence = collections.Counter(target_genes)
    relevant_genes_pd = pd.DataFrame(
        relevance_occurence.items(), columns=["Gene", "Occurence"]
    )
    relevant_genes_pd.to_csv(relevant_genes_path, sep="\t", index=True)
    relevant_genes_pd = relevant_genes_pd.sort_values("Occurence", ascending=False)
    relevant_genes_pd = relevant_genes_pd.reset_index(drop=True).head(15)
    gene_info = mygene.MyGeneInfo()
    ensembl_id_list = relevant_genes_pd["Gene"].tolist()
    ensembl_id_list = remove_ensembl_version(ensembl_id_list)
    retrieved_info = gene_info.querymany(ensembl_id_list, scopes="ensembl.gene")
    symbols = [info["symbol"] for info in retrieved_info]
    relevant_genes_pd["Symbol"] = symbols

    print(relevant_genes_pd.to_latex())

    return


def process_oncopole_wsi_data(
    model: torch.nn.Module, loader: Iterable, device: torch.device, artifacts_dir: str
) -> None:
    patches_list = []

    def coords(x):  # each wsi patch is 16 times bigger than the tiny image
        return x.cpu().item() // 16

    for step, batch in enumerate(
        tqdm(loader, desc="Computing gradcams", total=len(loader))
    ):
        matching_scores, _, grads, infos = compute_gradcam(
            model, batch, device, gradcam_on_rna=False
        )

        for matching_score, grad, info in zip(
            matching_scores.detach().cpu().numpy(), grads.detach().cpu().numpy(), infos
        ):
            wsi_name = info["patchFile"].split("_")[0]
            patch_coords = [coords(info[c]) for c in ["x1", "y1", "x2", "y2"]]
            patches_list.append(
                (wsi_name, info["patchFile"], patch_coords, matching_score, grad)
            )

    # Create a DataFrame from the patches_list
    patches_df = pd.DataFrame(
        patches_list,
        columns=["wsi_name", "patchFile", "patch_coords", "matching_score", "grad"],
    )

    # Group by wsi_name and aggregate the data
    grouped = (
        patches_df.groupby("wsi_name")
        .agg(
            {
                "patchFile": list,
                "patch_coords": list,
                "matching_score": list,
                "grad": list,
            }
        )
        .reset_index()
    )

    # for each wsi, open the wsi and save the matching scores and grads
    for _, row in tqdm(
        grouped.iterrows(),
        desc="Saving matching scores and gradients",
        total=len(grouped),
    ):
        wsi_name = row["wsi_name"]
        patch_files = row["patchFile"]
        patch_coords = row["patch_coords"]
        matching_scores = row["matching_score"]
        grads = row["grad"]
        save_wsi_matching_scores(
            matching_scores, grads, patch_files, patch_coords, wsi_name, artifacts_dir
        )

    return None


def save_wsi_matching_scores(
    matching_scores: List[float],
    grads: List[np.ndarray],
    patch_files: List[str],
    patch_coords: List[Tuple[int, int, int, int]],
    wsi_name: str,
    artifacts_dir: str,
) -> None:
    wsi_basename = os.path.basename(wsi_name.split("_")[0])
    wsi_basename = os.path.splitext(wsi_basename)[0]
    tiff_name = os.path.join("gbdata/tiny", wsi_basename + "_x0.625_z0.tif")
    tiny_wsi = pyvips.Image.new_from_file(tiff_name)
    wsi = tiny_wsi.numpy()  # h, w, c
    min_matching_score = min(matching_scores)
    max_matching_score = max(matching_scores)
    # normalize the matching scores to the range [0, 1]
    if max_matching_score == min_matching_score:  # this happens for some bad images
        matching_scores = [min_matching_score for _ in matching_scores]
    else:
        matching_scores = [
            (score - min_matching_score) / (max_matching_score - min_matching_score)
            for score in matching_scores
        ]
    # create a heatmap for the matching scores and overlap it with the wsi
    heatmap = np.zeros((wsi.shape[0], wsi.shape[1]))
    best_patch = None
    best_patch_score = -1
    best_patch_coords = patch_coords[0]
    best_grad = grads[0]
    for matching_score, patch_coord, patch_file, grad in zip(
        matching_scores, patch_coords, patch_files, grads
    ):
        x, y, w, h = patch_coord
        # fill the heatmap with the matching score
        heatmap[y:h, x:w] = matching_score

        # save the patch data with the highest matching score for gradcam visualization
        if matching_score > best_patch_score:
            best_patch_score = matching_score
            best_patch = patch_file
            best_patch_coords = patch_coord
            best_grad = grad

    # normalize the heatmap
    # overlap the heatmap with the wsi
    cmap = plt.get_cmap("plasma")
    heatmap = cmap(heatmap)
    heatmap = heatmap[:, :, :3]  # remove the alpha channel
    heatmap = heatmap * 255
    blend = wsi * 0.5 + heatmap * 0.5
    # draw a rectangle around the best patch
    x1, y1, x2, y2 = best_patch_coords
    cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 255, 0), 2)
    blend = blend.astype(np.uint8)
    # save the wsi
    blend_name = wsi_basename + "_scores.png"
    blend_name = os.path.join(artifacts_dir, blend_name)
    Image.fromarray(blend).save(blend_name)
    Image.fromarray(wsi).save(os.path.join(artifacts_dir, wsi_basename + ".png"))

    # Compute a heatmap over the gradient of the best patch
    best_grad_abs = np.abs(best_grad)
    # average the channels
    best_grad_abs = np.mean(best_grad_abs, axis=0)
    # Normalize the gradient to the range [0, 1]
    best_grad_normalized = (best_grad_abs - np.min(best_grad_abs)) / (
        np.max(best_grad_abs) - np.min(best_grad_abs)
    )
    # mask values smaller than a threshold
    best_grad_normalized[best_grad_normalized < 0.1] = 0
    best_grad_normalized = cmap(best_grad_normalized)[:, :, :3]
    best_grad_normalized = best_grad_normalized * 255
    # open the best patch image (it has the same size as the heatmap) as numpy array
    best_patch_img = np.array(Image.open(best_patch))
    # blend the heatmap with the best patch image
    blend_patch = best_patch_img * 0.25 + best_grad_normalized * 0.75
    blend_patch = Image.fromarray(blend_patch.astype(np.uint8))
    # get the basename of the best patch image
    wsi_patch_basename = os.path.basename(best_patch)
    wsi_patch_basename = os.path.splitext(wsi_patch_basename)[0]
    # save the blended image
    blend_patch.save(os.path.join(artifacts_dir, f"{wsi_patch_basename}_gradcam.png"))
    # save the original best patch image
    Image.fromarray(best_patch_img.astype(np.uint8)).save(
        os.path.join(artifacts_dir, wsi_patch_basename + "_original.png")
    )

    return None


def perform_attention_on_rna(
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
    gene_names: Dict,
    artifacts_dir: str,
) -> None:
    relevant_genes_path = os.path.join(artifacts_dir, "relevant_genes_val.csv")
    process_oncopole_rna_data(
        model, val_loader, device, gene_names, artifacts_dir, relevant_genes_path
    )

    return


def visualize(config: dict, gradcam_on_rna: bool) -> None:
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

    # Create datasets
    rna_dataset_params = config["rna_dataset_params"]
    img_dataset_params = config["img_dataset_params"]
    dataset = RNATumorDataset(rna_dataset_params, img_dataset_params, True)

    # Get validation dataset
    val_dataloader_params = config["val_dataloader_params"]
    val_dataset = copy.deepcopy(dataset)
    val_dataset.tumor_dataset.train = False
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **val_dataloader_params, shuffle=False
    )

    # Get gene names
    gene_names = dataset.rna_dataset.genes_per_cluster

    # Get image encoder
    img_model_params = config["img_model_params"]
    img_model = img_encoder.ImageEncoder(img_model_params)

    # Get rna encoder
    rna_model_params = config["rna_model_params"]
    rna_model_params["genes_per_cluster"] = (
        dataset.genes_per_cluster if hasattr(dataset, "genes_per_cluster") else None
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

    if gradcam_on_rna:
        print("======= Running gradcam on RNA data =======")
        perform_attention_on_rna(model, val_loader, device, gene_names, artifacts_dir)
    else:
        print("======= Running gradcam on WSI data =======")
        process_oncopole_wsi_data(model, val_loader, device, artifacts_dir)
