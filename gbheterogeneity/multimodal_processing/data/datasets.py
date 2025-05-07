import os
import random
import torch

import pandas as pd
import torch.utils.data as torch_data

import torchvision.transforms as torch_transforms
import gbheterogeneity.image_processing.data.datasets as tumor_dataset
import gbheterogeneity.rna_processing.data.datasets as rna_datasets

from typing import Dict, Tuple


class NoiseDataset(torch_data.Dataset):
    def __init__(self) -> None:
        self.length = 1000
        self.image_size = 256
        self.channels = 3
        self.num_clusters = 23
        self.num_genes_per_cluster = 800
        self.genes_per_cluster = self.get_genes_per_cluster()

    def get_genes_per_cluster(self):
        genes_per_cluster = {}
        for cluster in range(self.num_clusters):
            genes_per_cluster[str(cluster)] = [
                "gene_{}_{}".format(cluster, i)
                for i in range(self.num_genes_per_cluster)
            ]
        return genes_per_cluster

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict:
        patient_id = random.randrange(10)
        random_image = torch.rand(self.channels, self.image_size, self.image_size)
        random_image = (random_image - random_image.min()) / (
            random_image.max() - random_image.min() + 1e-9
        )

        rna_dict = {}
        for key, value in self.genes_per_cluster.items():
            rna_dict[key] = torch.rand(len(value))

        return random_image, rna_dict, patient_id


class TestDataset(rna_datasets.RNASeqOncopole):
    def __init__(
        self,
        oncopole_samples_directory: str,
        public_dataframe_path: str,
        gene_cluster_mapping_file: str,
        oncopole_rna_path: str,
    ) -> None:
        self.length = 1000
        self.image_size = 256
        self.channels = 3

        super().__init__(
            oncopole_samples_directory=oncopole_samples_directory,
            public_dataframe_path=public_dataframe_path,
            gene_cluster_mapping_file=gene_cluster_mapping_file,
            oncopole_rna_path=oncopole_rna_path,
        )

    def __getitem__(self, index: int) -> Tuple:
        random_image = torch.rand(self.channels, self.image_size, self.image_size)
        gene_file = self.gene_files[index]
        name = os.path.basename(gene_file)
        genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
        gene_values = self.get_values_by_genes(genes_df)
        return gene_values, random_image, name


class RNATumorDataset(torch_data.Dataset):
    def __init__(
        self, rna_dataset_params: Dict, img_dataset_params: Dict, train: bool = True
    ) -> None:
        self.rna_dataset = rna_datasets.RNASeqOncopole(
            oncopole_samples_directory=rna_dataset_params["oncopole_samples_directory"],
            tcga_split=rna_dataset_params["tcga_split"],
            gene_cluster_mapping_file=rna_dataset_params["gene_cluster_mapping_file"],
            oncopole_rna_path=rna_dataset_params["oncopole_rna_path"],
        )

        normalize = torch_transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        tumor_transforms = torch_transforms.Compose(
            [torch_transforms.ToTensor(), normalize]
        )
        self.tumor_dataset = tumor_dataset.TumorDataset(
            patchRootDir=img_dataset_params["path"],
            patchParams=img_dataset_params["patchParams"],
            train=train,
            transform=tumor_transforms,
            patientCSV=img_dataset_params["patients"],
            lineageCSV=img_dataset_params["lineage"],
        )

        self.genes_per_cluster = self.rna_dataset.genes_per_site

    def __len__(self):
        return len(self.tumor_dataset)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        image, image_info = self.tumor_dataset.__getitem__(index)
        rna_id = "SR" + str(image_info["patientID"]) + str(image_info["lineage"])
        try:
            gene_file = self.rna_dataset.gene_file_mapping[rna_id]
            genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
            gene_values = self.rna_dataset.get_values_by_genes(genes_df)
            return image, gene_values, image_info
        except KeyError:
            random_index = random.randrange(self.__len__())
            return self.__getitem__(random_index)


class RetrievalTumorDataset(torch_data.Dataset):
    def __init__(
        self, img_dataset_params: Dict, gene_file_mapping: Dict, train: bool = True
    ) -> None:
        normalize = torch_transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        tumor_transforms = torch_transforms.Compose(
            [torch_transforms.ToTensor(), normalize]
        )
        self.tumor_dataset = tumor_dataset.TumorDataset(
            patchRootDir=img_dataset_params["path"],
            patchParams=img_dataset_params["patchParams"],
            train=train,
            transform=tumor_transforms,
            patientCSV=img_dataset_params["patients"],
            lineageCSV=img_dataset_params["lineage"],
        )

        self.gene_file_mapping = gene_file_mapping

    def __len__(self):
        return len(self.tumor_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        image, image_info = self.tumor_dataset.__getitem__(index)
        patient_id = str(image_info["patientID"])
        lineage_id = patient_id + str(image_info["lineage"])
        rna_id = "SR" + lineage_id
        if rna_id in self.gene_file_mapping:
            return image, patient_id, lineage_id
        else:
            random_index = random.randrange(self.__len__())
            return self.__getitem__(random_index)
