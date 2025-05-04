import os
import pandas as pd

CLUSTERING_DIR = "gbdata/rna/gene_clustering/"

def main():
    kg_file = os.path.join(CLUSTERING_DIR,"kg.csv")
    if not os.path.exists(kg_file):
        raise FileNotFoundError(f"The kg.csv file should be downloaded to {CLUSTERING_DIR}.  See https://dataverse.harvard.edu/file.xhtml?fileId=6180620&version=2.1")
    rawKG = pd.read_csv(CLUSTERING_DIR + "kg.csv").rename(columns={"x_name": "gene_name"})
    filteredDF = rawKG[rawKG["relation"] == "protein_protein"]
    print(filteredDF.head())
    filteredDF.to_csv("ppi_full.csv")

    rawKG = pd.read_csv(kg_file).rename(columns={"x_name": "gene_name"})
    kgDF = rawKG[rawKG["relation"] == "protein_protein"].filter(
        [
            "x_index",
            "x_id",
            "x_type",
            "gene_name",
            "y_index",
            "y_id",
            "y_name",
        ]
    )

    rawRNA = pd.read_csv(
        CLUSTERING_DIR + "tcga_sample.tsv",
        skiprows=[0, 2, 3, 4, 5],
        sep="\t",
        usecols=["gene_id", "gene_name", "gene_type", "unstranded"],
    ).rename(columns={"unstranded": "rna_raw"})

    rnaDF = rawRNA[rawRNA["gene_type"] == "protein_coding"]
    rnaDF.drop(columns=["gene_type", "rna_raw"], inplace=True)

    combinedDF = pd.merge(kgDF, rnaDF, on="gene_name")
    print(combinedDF.head(20))
    combinedDF.to_csv(os.path.join(CLUSTERING_DIR, "ppi_filtered.csv"))

if __name__ == "__main__":
    main()
