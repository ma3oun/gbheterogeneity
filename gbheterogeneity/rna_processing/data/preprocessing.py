import json
import pandas as pd
from typing import Tuple, Dict, List


def remove_ensembl_version(ensembl_id_list: List[str]) -> List[str]:
    new_list = []
    dot = "."

    for ensembl_id in ensembl_id_list:
        if dot in ensembl_id:
            ensembl_id = ensembl_id.split(dot)[0]

        assert ensembl_id.startswith("ENSG")
        new_list.append(ensembl_id)

    return new_list


def get_common_genes(list_of_genes: List, contrastive_genes: List) -> List:
    common_genes = []
    for gene in list_of_genes:
        if gene in contrastive_genes:
            common_genes.append(gene)
    return common_genes


def filter_genes_per_cluster(genes_per_cluster: Dict, ensembl_list: List) -> Dict:
    filtered_genes_per_cluster = {}
    for chromosome, list_of_genes in genes_per_cluster.items():
        list_of_genes = get_common_genes(list_of_genes, ensembl_list)
        filtered_genes_per_cluster[chromosome] = list_of_genes
    return filtered_genes_per_cluster


def get_notation_mappings(common_genes: pd.DataFrame) -> Tuple:
    oncopole_genes_list = common_genes["Original_x"].tolist()
    tcga_genes_list = common_genes["Original_y"].tolist()
    ensembl_list = common_genes["Ensembl_only"].tolist()

    assert len(oncopole_genes_list) == len(tcga_genes_list)
    assert len(oncopole_genes_list) == len(ensembl_list)

    ensembl_to_tcga = {}
    ensembl_to_oncopole = {}
    for i, genes in enumerate(zip(oncopole_genes_list, tcga_genes_list, ensembl_list)):
        oncopole_gene, tcga_gen, ensembl_gen = genes
        ensembl_to_tcga[ensembl_gen] = tcga_gen
        ensembl_to_oncopole[ensembl_gen] = oncopole_gene

    return ensembl_list, ensembl_to_tcga, ensembl_to_oncopole


####################################


def get_tcga_genes(rna_path: str) -> pd.DataFrame:
    sample = pd.read_csv(rna_path, sep="\t", index_col=0)
    tcga_genes = sample.index.tolist()
    tcga_genes = pd.DataFrame(tcga_genes, columns=["Original"])

    return tcga_genes


def remove_ensembl_version_df(row: pd.DataFrame) -> bool:
    dot = "."
    ensembl_id = row["Original"]
    if dot in ensembl_id:
        new_ensembl_id = ensembl_id.split(dot)[0]
        assert new_ensembl_id.startswith("ENSG")

    return new_ensembl_id


def remove_version(tcga_genes: pd.DataFrame) -> pd.DataFrame:
    tcga_genes["Ensembl_only"] = tcga_genes.apply(
        lambda row: remove_ensembl_version_df(row), axis=1
    )

    return tcga_genes


def get_repeated_genes(tcga_genes: pd.DataFrame, column: str) -> pd.DataFrame:
    genes_without_version = tcga_genes[column]
    repeated_genes = tcga_genes[
        genes_without_version.isin(
            genes_without_version[genes_without_version.duplicated()]
        )
    ]
    repeated_genes = repeated_genes.sort_values(column)
    return repeated_genes


def get_one_out_two(indices: List) -> List:
    new_indices = []
    num_iters = len(indices) // 2

    for i in range(num_iters):
        index = indices[2 * i + 1]
        new_indices.append(index)

    return new_indices


def remove_duplicate_rows(tcga_genes: pd.DataFrame) -> pd.DataFrame:
    repeated_genes = get_repeated_genes(tcga_genes, "Ensembl_only")
    duplicated_genes = get_one_out_two(repeated_genes.index.tolist())
    tcga_genes = tcga_genes.drop(duplicated_genes)
    return tcga_genes


def remove_duplicated_gene(tcga_genes: List) -> List:
    dup_gene = "ENSG00000286105"
    dup_gene_df = tcga_genes["Ensembl_only"]  # type: pd.DataFrame
    dup_gene_df = dup_gene_df[dup_gene_df == dup_gene]
    dup_gene = dup_gene_df.index.tolist()
    tcga_genes = tcga_genes.drop(dup_gene)
    return tcga_genes


def get_standard_notation_tcga(tcga_rna_path: str) -> pd.DataFrame:
    tcga_genes = get_tcga_genes(tcga_rna_path)
    tcga_genes = remove_version(tcga_genes)
    tcga_genes = remove_duplicate_rows(tcga_genes)
    tcga_genes = remove_duplicated_gene(tcga_genes)

    return tcga_genes


####################################


def get_ensembl(index: str) -> str:
    assert isinstance(index, str)
    index = index.split("_")[0]
    assert index.startswith("ENSG")
    return index


def get_ensembl_notation(indices: List) -> List:
    new_indices = []
    for index in indices:
        new_index = get_ensembl(index)
        new_indices.append(new_index)

    return new_indices


def get_indices(counts_path: pd.DataFrame) -> List:
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    indices = counts.index.tolist()
    ensembl_indices = get_ensembl_notation(indices)
    assert len(indices) == len(ensembl_indices)
    assert len(set(indices)) == len(set(ensembl_indices))

    return indices, ensembl_indices


def get_standard_notation_oncopole(oncopole_rna_path: str) -> pd.DataFrame:
    indices, oncopole_ensembl_indices = get_indices(oncopole_rna_path)
    index_dict = {"Original": indices, "Ensembl_only": oncopole_ensembl_indices}
    oncopole_genes = pd.DataFrame(index_dict)
    return oncopole_genes


def open_mapping(mapping_file: str) -> Dict:
    with open(mapping_file) as json_file:
        genes_per_cluster = json.load(json_file)
        return genes_per_cluster


def translate_list(list_of_genes: List, mapping: Dict) -> List:
    new_list_of_genes = []
    for gene in list_of_genes:
        new_list_of_genes.append(mapping[gene])
    return new_list_of_genes


def get_mappings(
    filtered_genes_per_cluster: Dict, ensembl_to_tcga: Dict, ensembl_to_oncopole: Dict
) -> Tuple:
    mapping_ensembl_to_tcga = {}
    mapping_ensembl_to_oncopole = {}
    for chromosome, list_of_genes in filtered_genes_per_cluster.items():
        assert isinstance(list_of_genes, list)
        tcga_list_of_genes = translate_list(list_of_genes, ensembl_to_tcga)
        oncopole_list_of_genes = translate_list(list_of_genes, ensembl_to_oncopole)
        mapping_ensembl_to_tcga[chromosome] = tcga_list_of_genes
        mapping_ensembl_to_oncopole[chromosome] = oncopole_list_of_genes
        assert len(tcga_list_of_genes) == len(oncopole_list_of_genes)
    return mapping_ensembl_to_tcga, mapping_ensembl_to_oncopole


def build_mappings(
    tcga_rna_path: str, oncopole_rna_path: str, mapping_file: str
) -> Tuple:
    tcga_genes = get_standard_notation_tcga(tcga_rna_path)
    oncopole_genes = get_standard_notation_oncopole(oncopole_rna_path)
    genes_per_cluster = open_mapping(mapping_file)
    common_genes = oncopole_genes.merge(tcga_genes, on="Ensembl_only", how="inner")
    ensembl_list, ensembl_to_tcga, ensembl_to_oncopole = get_notation_mappings(
        common_genes
    )
    filtered_genes_per_chromosome = filter_genes_per_cluster(
        genes_per_cluster, ensembl_list
    )
    mapping_ensembl_to_tcga, mapping_ensembl_to_oncopole = get_mappings(
        filtered_genes_per_chromosome, ensembl_to_tcga, ensembl_to_oncopole
    )

    return mapping_ensembl_to_tcga, mapping_ensembl_to_oncopole
