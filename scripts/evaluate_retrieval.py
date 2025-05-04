from gbheterogeneity.multimodal_processing.rna_retrieval import evaluate

def main():
    config = {
        "rna_dataset_params": {
            "oncopole_samples_directory": "gbdata/rna/oncopole",
            "tcga_split": "gbdata/rna/test_split.tsv",
            "gene_cluster_mapping_file": "gbdata/rna/gene_clustering/ppi_cluster_to_ensembl_genes.json",
            "oncopole_rna_path": "gbdata/rna/oncopole_rna_raw.tsv",
        },
        "img_dataset_params": {
            "path": "gbdata/wsi",
            "patients": "gbdata/survival/patients.csv",
            "lineage": "gbdata/survival/lineage.csv",
            "patchParams": {
                "sequentialSampling": True,
                "patchH": 256,
                "patchW": 256,
                "minL": 0.05,
                "maxL": 0.8,
                "minS": 0.0,
                "maxS": 1.0,
                "colorMargin": 15,
                "tumorColor": 35.0,
                "neutralColor": 215.0,
                "tumorMinScore": 13107.2,
                "neutralMinScore": 42598.4,
                "maxPatchesPerImage": None
            }
        },
        "dataloader_params": {
            "batch_size": 32,
            "num_workers": 4
        },
        "val_dataloader_params": {
            "batch_size": 32,
            "num_workers": 4
        },
        "img_model_params": {
            "image_res": 256,
            "init_deit": False,
            "freeze_vision_encoder": False,
            "freeze_projection_heads": False,
            "vision_width": 768,
            "embed_dim": 256
        },
        "rna_model_params": {
            "embedding_dim": 128,
            "dropout_prob": 0.5,
            "num_heads": 8,
            "projection_dim": 64
        },
        "multimodal_model_params": {
            "co_attention_dim": 768,
            "image_emb_size": 768,
            "rna_emb_size": 128,
            "projection_size": 128,
            "num_heads": 8,
            "freeze_img_model": False,
            "freeze_rna_model": False,
        },
        "logger_params": {
            "experiment_name": "paper_v2_RNA_retrieval_best",
            "experiment_dir": "output/mlruns",
            
        },
        "device": "cuda",
        "manual_seed": 42,
        "pretrained_path": "trained_models/multimodal_best.bin"
    }
    evaluate(config)

if __name__ == "__main__":
    main()