import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix
from typing import Dict
from sknetwork import sknetwork as skn

CLUSTERING_DIR = "gbdata/rna/gene_clustering/"

def clusterizePPI(filteredPPI_File:str,verbose:bool=False)->Dict:
    """
    Perform PPI graph clustering using different methods.

    Parameters
    ----------
    filteredPPI_File : str
        Path to the filtered PPI dataset.
    verbose: bool, optional
        Show intermediate output info

    Returns
    -------
    Dataframe, Genes with assigned label.

    """
    # Read filtered PPI dataset

    ppiDF = pd.read_csv(
        filteredPPI_File,
        usecols=[
            "x_index",
            "x_id",
            "gene_name",
            "gene_id",
            "y_index",
            "y_id",
            "y_name",
        ],
    )

    # PPI graph
    ppi = nx.from_pandas_edgelist(ppiDF, source="x_index", target="y_index")

    components = list(nx.connected_components(ppi))
    connected_subgraphs = [ppi.subgraph(c).copy() for c in components]

    if verbose:
        print(f"PPI graph has {len(connected_subgraphs)} connected sub-graphs")

    ppi_main = connected_subgraphs[0]
    ppiDF.set_index("x_index", drop=False, inplace=True)
    for otherComponents in components[1:]:
        ppiDF.drop(otherComponents, inplace=True)
    ppiDF.set_index("y_index", drop=False, inplace=True)

    for otherComponents in components[1:]:
        try:
            ppiDF.drop(otherComponents, inplace=True)
        except KeyError:
            print(f"Skipping {otherComponents}")
    ppi_main = nx.from_pandas_edgelist(ppiDF, source="x_index", target="y_index")

    graphNodes = {int(n) for n in ppi_main.nodes()}  # set of unique nodes
    if verbose:
        print(f"Nodes in graph: {len(graphNodes)}")

    louvain_labels_file = os.path.join(CLUSTERING_DIR,"louvain_labels.npz")
    louvain_membership_file = os.path.join(CLUSTERING_DIR,"louvain_membership.npz")

    if os.path.exists(louvain_labels_file) and os.path.exists(louvain_membership_file):
        input(
            "Graph clustering files already exists. Press enter to continue to replace existing files or ctrl+c to quit."
        )
        os.remove(louvain_labels_file)
        os.remove(louvain_membership_file)
    louvain = skn.clustering.Louvain()
    louvain_labels = louvain.fit_predict(csr_matrix(nx.adjacency_matrix(ppi_main)))
    membershipMatrix = louvain.aggregate_
    np.savez_compressed(louvain_labels_file, louvain_labels)
    np.savez_compressed(louvain_membership_file, membershipMatrix.todense())

    labels = np.load(louvain_labels_file, allow_pickle=True)["arr_0"]
    hist, _ = np.histogram(labels, list(range(12)))
    if verbose:
        print(f"Labels histogram:\n{hist}")

    index2label = dict()
    for node_idx, node in enumerate(ppi_main.nodes()):
        index2label[node] = labels[node_idx]

    return ppiDF,index2label


def assignLabels(ppi: pd.DataFrame, nodeLabels: dict) -> pd.DataFrame:
    """
    Assigns cluster labels to genes based on the provided node labels.

    This function takes a DataFrame containing PPI (Protein-Protein Interaction) data and a dictionary
    mapping node indices to cluster labels. It assigns the cluster labels to the corresponding genes
    and returns a DataFrame with the gene indices, gene names, and cluster labels.

    Parameters
    ----------
    ppi : pd.DataFrame
        DataFrame containing PPI data with columns 'x_index', 'gene_name', 'y_index', and 'y_name'.
    nodeLabels : dict
        Dictionary mapping node indices to cluster labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'gene_index', 'gene_name', and 'cluster' containing the assigned cluster labels.

    Notes
    -----
    - If a node index appears in both 'x_index' and 'y_index', the function will use the label from 'x_index'.
    - If a node index is not found in either 'x_index' or 'y_index', a warning message is printed.
    - Cluster labels greater than or equal to 6 are mapped to 6.
    """
    
    geneLabelsDict = dict()
    x_data = set(zip(ppi["x_index"].to_list(), ppi["gene_name"].to_list()))
    x_dict = {idx: name for idx, name in x_data}

    y_data = set(zip(ppi["y_index"].to_list(), ppi["y_name"].to_list()))
    y_dict = {idx: name for idx, name in y_data}
    for node, label in nodeLabels.items():
        if node in x_dict.keys():
            if node in geneLabelsDict.keys():
                currentLabel = geneLabelsDict[node]
                if currentLabel != label:
                    print(f"Node: {node} has switched from {currentLabel} to {label}")
            geneLabelsDict[node] = dict(
                gene_idx=node, gene_name=x_dict[node], label=label
            )
        elif node in y_dict.keys():
            if node in geneLabelsDict.keys():
                currentLabel = geneLabelsDict[node]
                if currentLabel != label:
                    print(f"Node: {node} has switched from {currentLabel} to {label}")
            geneLabelsDict[node] = dict(
                gene_idx=node, gene_name=y_dict[node], label=label
            )
        else:
            print(f"Node {node} cannot be found in either x_index nor y_index")

    dataFrameData = dict(gene_index=[], gene_name=[], cluster=[])
    for geneData in geneLabelsDict.values():
        dataFrameData["gene_index"].append(geneData["gene_idx"])
        dataFrameData["gene_name"].append(geneData["gene_name"])
        geneDataLabel = geneData["label"]
        if geneDataLabel >= 6:
            geneDataLabel = 6

        dataFrameData["cluster"].append(geneDataLabel)

    return pd.DataFrame(dataFrameData)

def filter_to_tcga(clusters_df,tcga_sample_file):
    tcga_data = pd.read_csv(tcga_sample_file,skiprows=[0,2,3,4,5],sep="\t",usecols=["gene_id", "gene_name", "gene_type", "unstranded"])
    tcga_data = tcga_data[tcga_data["gene_type"] == "protein_coding"]
    tcga_data.drop(columns=["gene_type", "unstranded", "gene_type"], inplace=True)
    return pd.merge(clusters_df, tcga_data, on="gene_name")

def group_genes_by_cluster(ppi_clusters,symbols_to_ensembl_mapping: dict):
    df = ppi_clusters.groupby("cluster")["gene_name"].apply(list)
    df = df.apply(lambda x: [symbols_to_ensembl_mapping[gene] for gene in x if gene in symbols_to_ensembl_mapping.keys()])
    return df

def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file,"r") as f:
        for line in f:
            gene_name, gene_id = line.strip().split(",")
            mapping[gene_name] = gene_id
    return mapping

def main():
    ppi_filtered_file = os.path.join(CLUSTERING_DIR,"ppi_filtered.csv")
    ppiDF,mapping = clusterizePPI(ppi_filtered_file,verbose=True)
    symbol_to_ensembl_mapping = load_mapping(os.path.join(CLUSTERING_DIR,"symbol_to_ensembl_mapping.csv"))
    ppi_clusters = assignLabels(ppiDF, mapping)

    tcga_file = os.path.join(CLUSTERING_DIR,"tcga_sample.tsv")
    ppi_clusters = filter_to_tcga(ppi_clusters,tcga_file)

    ppi_clusters.to_csv(os.path.join(CLUSTERING_DIR,"ppi_clusters.csv"))
    grouped_genes = group_genes_by_cluster(ppi_clusters,symbol_to_ensembl_mapping)
    grouped_genes.to_json(os.path.join(CLUSTERING_DIR,"ppi_cluster_to_ensembl_genes.json"), orient="index",indent=4)

if __name__ == "__main__":
    main()
