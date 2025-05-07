from gbheterogeneity.image_processing.train import train


def main() -> None:
    config = {
        "dataset_params": {
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
        "val_dataset_params": {"something": -1},
        "dataloader_params": {"batch_size": 64, "num_workers": 4},
        "val_dataloader_params": {"batch_size": 64, "num_workers": 4},
        "model_params": {
            "image_res": 256,
            "init_deit": True,
            "freeze_vision_encoder": False,
            "freeze_projection_heads": False,
            "vision_width": 768,
            "embed_dim": 256,
            "optimizer_name": "AdamWOptimizer",
            "optimizer_params": {"learning_rate": 0.0001, "weight_decay": 0.02},
            "scheduler_name": "CosineScheduler",
            "scheduler_params": {
                "epochs": 25,
                "min_lr": 0.00001,
                "decay_rate": 1,
                "warmup_lr": 0.00001,
                "warmup_epochs": 6,
            },
        },
        "logger_params": {
            "experiment_name": "paper_v2_wsi",
            "experiment_dir": "output/mlruns",
        },
        "logging_params": {"log_every_n_steps": 10, "log_n_last_models": 5},
        "trainer": "trainer",
        "trainer_params": {
            "margin": 0.0001,
            "max_epochs": 25,  # same as scheduler
            "temp": 0.005,
            "coeff_contrastive_loss": 1.0,
            "warmup_epochs": 6,  # same as scheduler
            "device": "cuda",
            "manual_seed": 88,
        },
    }
    train(config)


if __name__ == "__main__":
    main()
