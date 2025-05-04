import math
import numpy as np
import torch

import torch.utils.data as data
import torch.distributed as dist

import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools
import gbheterogeneity.utils.pytorch.simple_distributed_trainer as simple_trainer
from gbheterogeneity.utils.pytorch.selector import (
    OnlineTripletLoss,
    HardestNegativeTripletSelector,
)

from typing import Dict, Tuple, List
from tqdm import tqdm


class Trainer(simple_trainer.SimpleDistributedTrainer):
    def __init__(
        self,
        start_epoch: int = 0,
        max_epochs: int = 50,
        temp: float = 0.005,
        coeff_contrastive_loss: float = 1.0,
        warmup_epochs: int = 20,
        device: torch.device = None,
        dist_params: Dict = {},
        log_every_n_steps: int = 50,
        log_n_last_models: int = 5,
        gradient_accumulation_steps: int = 1,
        save_directory: str = "",
        resume_file: str = "",
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        margin: float = None,
        **kwargs,
    ) -> None:

        super().__init__(
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            device=device,
            dist_params=dist_params,
            log_every_n_steps=log_every_n_steps,
            log_n_last_models=log_n_last_models,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_directory=save_directory,
            resume_file=resume_file,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.margin = margin
        self.warmup_epochs = warmup_epochs

        self.temp = temp
        self.coeff_contrastive_loss = coeff_contrastive_loss
        self.loss_contrastive_key = "tripletL"

    def data_to_device(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        return images

    def compute_contrastive_loss(
        self, projections: torch.Tensor, labels: List[int]
    ) -> torch.Tensor:
        loss_fn = OnlineTripletLoss(
            self.margin, HardestNegativeTripletSelector(margin=self.margin)
        )
        loss, _ = loss_fn(projections, labels)

        return loss

    def compute_loss(self, projections: torch.Tensor, labels: List[int]) -> Dict:
        loss_contrastive = self.compute_contrastive_loss(projections, labels)
        loss = self.coeff_contrastive_loss * loss_contrastive
        loss_dict = {self.loss_key: loss, self.loss_contrastive_key: loss_contrastive}

        return loss_dict

    def save_logs(self, loss_dict: Dict, prefix: str) -> None:
        if (
            self.global_step % self.log_every_n_steps == 0
            and dst_tools.is_main_process()
        ):
            if prefix == "Train":
                learning_rate = torch.tensor(self.optimizer.param_groups[0]["lr"])
                mlflow_logging.log_metric(
                    key="Learning rate", value=learning_rate, step=self.global_step
                )
            mlflow_logging.log_metrics(loss_dict, self.global_step, prefix=prefix)

    def training_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        images, imagesData = batch
        labels = imagesData["lineageLabel"]

        images = self.data_to_device(images)
        _, image_features = model(images)
        loss_dict = self.compute_loss(image_features, labels)
        loss = loss_dict[self.loss_key]
        self.save_logs(loss_dict, "Train")

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if (
            self.epoch == 0
            and step % self.step_size == 0
            and step <= self.warmup_iterations
        ):
            self.scheduler.step(step // self.step_size)

        self.global_step += 1

        return loss_dict

    def validation_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        images, imagesData = batch
        labels = imagesData["lineageLabel"]
        images = self.data_to_device(images)
        _, image_features = model(images)
        loss_dict = self.compute_loss(image_features, labels)
        self.save_logs(loss_dict, "Validation")

        return loss_dict

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader = None,
    ) -> None:
        n_training_batches, n_validation_batches = self.compute_n_batches(
            train_loader, val_loader
        )
        self.step_size = math.floor(n_training_batches / self.warmup_epochs)
        self.warmup_iterations = self.warmup_epochs * self.step_size
        model, start_epoch = self.resume_training_if_provided(model)

        if self.dist_params["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.dist_params["gpu"]]
                )

        self.optimizer.zero_grad()

        with tqdm(range(start_epoch, self.end_epoch), desc="Train", position=0) as bar:
            for epoch_id in bar:
                # Setting epoch id and loss accumulators
                self.epoch = epoch_id
                train_outputs, val_outputs = [], []
                isBest = False

                # Update scheduler
                if epoch_id > 0:
                    self.scheduler.step(epoch_id + self.warmup_epochs)

                # Training the model
                model.train()
                if self.dist_params["distributed"]:
                    train_loader.sampler.set_epoch(epoch_id)
                for step, batch in tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc="Epoch",
                    leave=False,
                    position=1,
                ):
                    loss_dict = self.training_step(model, batch, step)
                    train_outputs.append(loss_dict)
                    self.print_status("Train", step, n_training_batches, loss_dict, bar)
                self.training_epoch_end(train_outputs)

                # Evaluating the model
                if val_loader is not None:
                    torch.set_grad_enabled(False)
                    model.eval()
                    with tqdm(
                        enumerate(val_loader),
                        total=len(val_loader),
                        desc="Validation",
                        leave=False,
                        position=2,
                    ) as bar_val:
                        for step, batch in bar_val:
                            loss_dict_val = self.validation_step(model, batch, step)
                            val_outputs.append(loss_dict_val)
                            self.print_status(
                                "Validation",
                                step,
                                n_validation_batches,
                                loss_dict_val,
                                bar_val,
                            )
                    self.validation_epoch_end(val_outputs)

                    mean_val_loss = np.mean(
                        [l[self.loss_key].item() for l in val_outputs]
                    )
                    if (
                        self.best_model_score is None
                    ) or mean_val_loss < self.best_model_score:
                        isBest = True
                        self.best_model_score = mean_val_loss
                    else:
                        isBest = False

                    torch.set_grad_enabled(True)

                if self.dist_params["distributed"]:
                    dist.barrier()

                self.save_model(model, loss_dict, isBest)
