import os
import glob
import torch
import json

import numpy as np
import pandas as pd
import torch.utils.data as data

from typing import Tuple, List, Dict
from .preprocessing import build_mappings


class NoiseDatasetAttention(data.Dataset):
    def __init__(self) -> None:
        self.length = 10000
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
        rna_dict = {}
        for key, value in self.genes_per_cluster.items():
            rna_dict[key] = torch.rand(len(value))

        return rna_dict


class NoiseDatasetAE(data.Dataset):
    def __init__(self) -> None:
        self.length = 1000
        self.num_genes = 20000

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        rna_expression = torch.rand(self.num_genes)
        return rna_expression


class RNASeq(data.Dataset):
    def __init__(
        self, tcga_split: str, gene_cluster_mapping_file: str, oncopole_rna_path: str
    ) -> None:
        self.tcga_split = tcga_split
        self.gene_cluster_mapping_file = gene_cluster_mapping_file
        self.oncopole_rna_path = oncopole_rna_path

        self.gene_files = self.get_gene_files()
        self.genes_per_cluster = self.get_genes_per_cluster()
        self.genes_per_site = self.get_genes_per_site()

    def get_gene_files(self) -> List:
        files = pd.read_csv(self.tcga_split, sep="\t", index_col=0)
        files = files["Sample paths"].tolist()
        return files

    def get_genes_per_cluster(self):
        mapping_ensembl_to_public, _ = build_mappings(
            self.gene_files[0], self.oncopole_rna_path, self.gene_cluster_mapping_file
        )
        return mapping_ensembl_to_public

    def get_genes_per_site(self) -> Dict:
        gene_values = self.__getitem__(0)[0]
        genes_per_site = {}
        for key, value in gene_values.items():
            genes_per_site[key] = value.shape[0]

        return genes_per_site

    def get_values_by_genes(self, dataframe: pd.DataFrame) -> Dict:
        gene_values = {}
        for key, list_of_genes in self.genes_per_cluster.items():
            gene_cluster_df = dataframe.loc[list_of_genes]
            gene_cluster_value = gene_cluster_df.values  # numpy array
            gene_values[key] = np.squeeze(np.copy(gene_cluster_value), axis=1)
        return gene_values

    def __len__(self) -> int:
        return len(self.gene_files)

    def __getitem__(self, index: int) -> Tuple:
        gene_file = self.gene_files[index]
        name = os.path.basename(gene_file)
        genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
        gene_values = self.get_values_by_genes(genes_df)
        return gene_values, name


class LabeledRNASeq(RNASeq):
    def __init__(
        self,
        tcga_split: str,
        gene_cluster_mapping_file: str,
        oncopole_rna_path: str,
        site_mapping_path: str,
    ) -> None:
        self.tcga_split = tcga_split
        self.site_mapping_path = site_mapping_path
        self.dataframe = self.get_dataframe()
        self.times = self.get_survival_time()
        self.sites = self.get_site()
        self.site_mapping = self.get_site_mapping()
        self.inverse_site_mapping = self.get_inverse_site_mapping()

        super().__init__(
            tcga_split=tcga_split,
            gene_cluster_mapping_file=gene_cluster_mapping_file,
            oncopole_rna_path=oncopole_rna_path,
        )

    def get_site_mapping(self) -> Dict:
        with open(self.site_mapping_path) as json_file:
            mapping = json.load(json_file)
        return mapping

    def get_inverse_site_mapping(self) -> Dict:
        mapping = {}
        for key, values in self.site_mapping.items():
            mapping[values] = key
        return mapping

    def get_dataframe(self) -> List:
        dataframe = pd.read_csv(self.tcga_split, sep="\t", index_col=0)
        return dataframe

    def get_gene_files(self) -> List:
        files = self.dataframe["Sample paths"].tolist()
        return files

    def get_survival_time(self) -> List:
        times = self.dataframe["time"].tolist()
        return times

    def get_site(self) -> List:
        sites = self.dataframe["primary_site"].tolist()
        return sites

    def __getitem__(self, index: int) -> Tuple:
        gene_file = self.gene_files[index]
        time = self.times[index]
        site = self.sites[index]
        site = self.site_mapping[site]
        name = os.path.basename(gene_file)
        genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
        gene_values = self.get_values_by_genes(genes_df)
        return gene_values, name, time, site


class RNASeqOncopole(data.Dataset):
    def __init__(
        self,
        oncopole_samples_directory: str,
        tcga_split: str,
        gene_cluster_mapping_file: str,
        oncopole_rna_path: str,
    ) -> None:
        self.oncopole_samples_directory = oncopole_samples_directory
        self.tcga_split = tcga_split
        self.gene_cluster_mapping_file = gene_cluster_mapping_file
        self.oncopole_rna_path = oncopole_rna_path

        self.gene_files = self.get_gene_files()
        self.gene_file_mapping = self.get_gene_file_mapping()
        self.public_gene_files = self.get_public_gene_files()
        self.genes_per_cluster = self.get_genes_per_cluster()
        self.genes_per_site = self.get_genes_per_site()

    def get_gene_files(self) -> List:
        query = os.path.join(self.oncopole_samples_directory, "*.tsv")
        files = glob.glob(query)
        return files

    def get_gene_file_mapping(self) -> Dict:
        gene_file_mapping = {}

        for gene_file in self.gene_files:
            name = os.path.basename(gene_file)
            name = name.split(".")[0]
            gene_file_mapping[name] = gene_file

        return gene_file_mapping

    def get_public_gene_files(self) -> List:
        files = pd.read_csv(self.tcga_split, sep="\t", index_col=0)
        files = files["Sample paths"].tolist()
        return files

    def get_genes_per_cluster(self):
        mapping_ensembl_to_public, _ = build_mappings(
            self.public_gene_files[0],
            self.oncopole_rna_path,
            self.gene_cluster_mapping_file,
        )
        return mapping_ensembl_to_public

    def get_genes_per_site(self) -> Dict:
        gene_values = self.__getitem__(0)[0]
        genes_per_site = {}
        for key, value in gene_values.items():
            genes_per_site[key] = value.shape[0]

        return genes_per_site

    def get_values_by_genes(self, dataframe: pd.DataFrame) -> Dict:
        gene_values = {}
        for key, list_of_genes in self.genes_per_cluster.items():
            gene_cluster_df = dataframe.loc[list_of_genes]
            gene_cluster_value = gene_cluster_df.values  # numpy array
            gene_values[key] = np.squeeze(np.copy(gene_cluster_value), axis=1)
        return gene_values

    def get_patient_id(self, name: str) -> str:
        name = name.split(".")[0]
        assert name.startswith("SR")
        return name[2]

    def __len__(self) -> int:
        return len(self.gene_files)

    def __getitem__(self, index: int) -> Tuple:
        gene_file = self.gene_files[index]
        name = os.path.basename(gene_file)
        patient_id = self.get_patient_id(name)
        genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
        gene_values = self.get_values_by_genes(genes_df)
        return gene_values, patient_id


class RetrievalRNASeq(RNASeqOncopole):
    def __init__(
        self,
        oncopole_samples_directory: str,
        tcga_split: str,
        gene_cluster_mapping_file: str,
        oncopole_rna_path: str,
    ) -> None:
        super().__init__(
            oncopole_samples_directory=oncopole_samples_directory,
            tcga_split=tcga_split,
            gene_cluster_mapping_file=gene_cluster_mapping_file,
            oncopole_rna_path=oncopole_rna_path,
        )

    def get_patient_id(self, name: str) -> str:
        name = name.split(".")[0]
        assert name.startswith("SR")
        return name[2]

    def get_lineage_id(self, name: str) -> str:
        name = name.split(".")[0]
        assert name.startswith("SR")
        return name[2:4]

    def __getitem__(self, index: int) -> Tuple:
        gene_file = self.gene_files[index]
        name = os.path.basename(gene_file)
        patient_id = self.get_patient_id(name)
        lineage_id = self.get_lineage_id(name)
        genes_df = pd.read_csv(gene_file, sep="\t", index_col=0)
        gene_values = self.get_values_by_genes(genes_df)

        return gene_values, patient_id, lineage_id


def get_rna_attention_data(
    dataset_params: Dict,
    dataloader_params: Dict,
    num_gpus: int,
    global_rank: int,
    train: bool,
) -> data.DataLoader:
    dataset = RNASeq(
        tcga_split=dataset_params["tcga_split"],
        gene_cluster_mapping_file=dataset_params["gene_cluster_mapping_file"],
        oncopole_rna_path=dataset_params["oncopole_rna_path"],
    )
    genes_per_cluster = dataset.genes_per_cluster
    if train and num_gpus:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=global_rank, shuffle=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
        )
    elif train and num_gpus == 0:
        loader = torch.utils.data.DataLoader(dataset, **dataloader_params, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, **dataloader_params, shuffle=False
        )
    loader.genes_per_cluster = genes_per_cluster
    return loader


def get_rna_attention_labeled_data(
    dataset_params: Dict,
    dataloader_params: Dict,
    num_gpus: int,
    global_rank: int,
    train: bool,
) -> data.DataLoader:
    dataset = LabeledRNASeq(
        tcga_split=dataset_params["tcga_split"],
        gene_cluster_mapping_file=dataset_params["gene_cluster_mapping_file"],
        oncopole_rna_path=dataset_params["oncopole_rna_path"],
        site_mapping_path=dataset_params["site_mapping_path"],
    )
    genes_per_cluster = dataset.genes_per_cluster
    inverse_site_mapping = dataset.inverse_site_mapping
    if train and num_gpus:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=global_rank, shuffle=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
        )
    elif train and num_gpus == 0:
        loader = torch.utils.data.DataLoader(dataset, **dataloader_params, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, **dataloader_params, shuffle=False
        )
    loader.genes_per_cluster = genes_per_cluster
    loader.inverse_site_mapping = inverse_site_mapping
    return loader


def get_rna_attention_data_oncopole(
    dataset_params: Dict,
    dataloader_params: Dict,
    num_gpus: int,
    global_rank: int,
    train: bool,
) -> data.DataLoader:
    dataset = RNASeqOncopole(
        oncopole_samples_directory=dataset_params["oncopole_samples_directory"],
        tcga_split=dataset_params["tcga_split"],
        gene_cluster_mapping_file=dataset_params["gene_cluster_mapping_file"],
        oncopole_rna_path=dataset_params["oncopole_rna_path"],
    )
    genes_per_cluster = dataset.genes_per_site
    if train and num_gpus:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=global_rank, shuffle=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params,
            shuffle=False,
            pin_memory=True,
            sampler=sampler,
        )
    elif train and num_gpus == 0:
        loader = torch.utils.data.DataLoader(dataset, **dataloader_params, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, **dataloader_params, shuffle=False
        )
    loader.genes_per_cluster = genes_per_cluster
    return loader
