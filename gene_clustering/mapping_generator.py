import mygene
import pandas as pd
import os

CLUSTERING_DIR = "gbdata/rna/gene_clustering"

def main():
    """
    This script generates a mapping from gene symbols to Ensembl IDs."""
    
    ppi_df = pd.read_csv(os.path.join(CLUSTERING_DIR,"ppi_filtered.csv"),usecols=[
            "gene_name",
            "gene_id",
            "y_name",
        ])
    df = ppi_df[["gene_name","gene_id"]]
    df["gene_id"] = df["gene_id"].apply(lambda x: x.split(".")[0])
    x_mapping = df.set_index("gene_name")["gene_id"].to_dict()
    y_genes = ppi_df["y_name"].unique().tolist()
    y_genes_unique = [gene for gene in y_genes if gene not in x_mapping.keys()]
    problematic_genes = []
    mg = mygene.MyGeneInfo()
    y_mapping = mg.querymany(y_genes_unique, scopes="symbol", fields="ensembl.gene", species="human", returnall=True)

    no_gene_in_ensembl = []
    no_ensembl = []
    no_query = []
    for item in y_mapping['out']:
        if 'query' in item:
            if 'ensembl' in item:
                if 'gene' in item['ensembl']:
                    x_mapping[item['query']] = item['ensembl']['gene']
                else:
                    problematic_genes.append(item['query'])
                    no_gene_in_ensembl.append(item['query'])
            else:
                problematic_genes.append(item['query'])
                no_ensembl.append(item['query'])
        else:
            print(item)
            no_query.append(item)
    print(f"no_gene_in_ensembl: {no_gene_in_ensembl} (total: {len(no_gene_in_ensembl)})\n\n")
    print(f"no_ensembl: {no_ensembl} (total: {len(no_ensembl)})\n\n")
    print(f"no_query: {no_query} (total: {len(no_query)})")

    with open(os.path.join(CLUSTERING_DIR,"symbol_to_ensembl_mapping.csv"),"w") as f:
        for key,value in x_mapping.items():
            f.write(f"{key},{value}\n")

if __name__ == "__main__":
    main()