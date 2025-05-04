from gbheterogeneity.rna_processing.train import train

def main() -> None:
    config = {
        "dataset_name": "AttentionRNAData",
        "val_dataset_name": "AttentionRNAData",
        "dataset_params": {
            "tcga_split": "gbdata/rna/train_split.tsv",
            "gene_cluster_mapping_file": "gbdata/rna/gene_clustering/ppi_cluster_to_ensembl_genes.json",
            "oncopole_rna_path": "gbdata/rna/oncopole_rna_raw.tsv"
        },
        "val_dataset_params": {
            "tcga_split": "gbdata/rna/val_split.tsv",
            "gene_cluster_mapping_file": "gbdata/rna/gene_clustering/ppi_cluster_to_ensembl_genes.json",
            "oncopole_rna_path": "gbdata/rna/oncopole_rna_raw.tsv"
        },
        "dataloader_params": {
            "batch_size": 32,
            "num_workers": 16
        },
        "val_dataloader_params": {
            "batch_size": 32,
            "num_workers": 16
        },
        "model_name": "AttentionRNA",
        "model_params": {
            "embedding_dim": 128,
            "dropout_prob": 0.5,
            "num_heads": 8,
            "projection_dim": 64
        },
        "optimizer_name": "AdamWOptimizer",
        "optimizer_params": {
            "learning_rate": 0.0001,
            "weight_decay": 0.02
        },
        "scheduler_name": "CosineScheduler",
        "scheduler_params": {
            "epochs": 20,
            "min_lr": 0.00001,
            "decay_rate": 1,
            "warmup_lr": 0.00001,
            "warmup_epochs": 6
        },
        "logger_params": {
            "experiment_name": "paper_v2_rna_ppi",
            "experiment_dir": "output/mlruns"
        },
        "logging_params": {
            "log_every_n_steps": 10,
            "log_n_last_models": 1
        },
        "trainer_name": "AttentionTrainer",
        "trainer_params": {
            "max_epochs": 20,  # same as scheduler
            "temp": 0.005,
            "mask_prob": 0.30,
            "coeff_reconstruction_loss": 1.0,
            "coeff_contrastive_loss": 1.0,
            "warmup_epochs": 6,  # same as scheduler
            "device": "cuda",
            "manual_seed": 42
        },
    }
    train(config)


if __name__ == "__main__":
    main()
