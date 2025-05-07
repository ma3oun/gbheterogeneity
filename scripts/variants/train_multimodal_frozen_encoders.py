from gbheterogeneity.multimodal_processing.train import train


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
                "maxPatchesPerImage": None,
            },
        },
        "dataloader_params": {"batch_size": 64, "num_workers": 4},
        "val_dataloader_params": {"batch_size": 64, "num_workers": 4},
        "img_model_params": {
            "image_res": 256,
            "init_deit": False,
            "freeze_vision_encoder": False,
            "freeze_projection_heads": False,
            "vision_width": 768,
            "embed_dim": 256,
            "img_pretrained_path": "trained_models/wsi_encoder.bin",
        },
        "rna_model_params": {
            "embedding_dim": 128,
            "dropout_prob": 0.5,
            "num_heads": 8,
            "projection_dim": 64,
            "rna_pretrained_path": "trained_models/rna_encoder_ppi.bin",
        },
        "multimodal_model_params": {
            "co_attention_dim": 768,
            "image_emb_size": 768,
            "rna_emb_size": 128,
            "projection_size": 128,
            "num_heads": 8,
            "freeze_img_model": True,
            "freeze_rna_model": True,
        },
        "optimizer_name": "AdamWOptimizer",
        "optimizer_params": {"learning_rate": 0.0001, "weight_decay": 0.02},
        "scheduler_name": "CosineScheduler",
        "scheduler_params": {
            "epochs": 10,
            "min_lr": 0.00001,
            "decay_rate": 1,
            "warmup_lr": 0.00001,
            "warmup_epochs": 6,
        },
        "logger_params": {
            "experiment_name": "paper_v2_multimodal_frozen_encoders",
            "experiment_dir": "output/mlruns",
        },
        "logging_params": {"log_every_n_steps": 10, "log_n_last_models": 4},
        "trainer_params": {
            "max_epochs": 10,
            "temp": 0.005,
            "coeff_contrastive_loss": 1.0,
            "coeff_matching_loss": 1.0,
            "coeff_reconstruction_loss": 1.0,
            "warmup_epochs": 6,
            "device": "cuda",
            "manual_seed": 42,
        },
    }
    train(config)


if __name__ == "__main__":
    main()
