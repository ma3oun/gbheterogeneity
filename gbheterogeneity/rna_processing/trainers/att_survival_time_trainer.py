import os
import torch
import pickle

import torch.utils.data as data
import torch.distributed as dist

import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools
import gbheterogeneity.utils.pytorch.simple_distributed_trainer as simple_trainer

from typing import Dict, Tuple, List


class AttentionTrainerSurvivalTime(simple_trainer.SimpleDistributedTrainer):
    def __init__(
        self,
        start_epoch: int = 0,
        max_epochs: int = 50,
        coeff_loss: float = 1.0,
        device: torch.device = None,
        dist_params: Dict = {},
        log_every_n_steps: int = 50,
        log_n_last_models: int = 5,
        gradient_accumulation_steps: int = 1,
        save_directory: str = "",
        resume_file: str = "",
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
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

        self.coeff_loss = coeff_loss
        self.loss_l1_key = "loss_l1"
        self.loss_l2_key = "loss_l2"
        self.accuracy_3_months_str = "accuracy_3_months"
        self.accuracy_6_months_str = "accuracy_6_months"
        self.accuracy_12_months_str = "accuracy_12_months"

    def target_to_device(self, target: torch.Tensor) -> torch.Tensor:
        return target.float().to(self.device)

    def data_to_device(
        self, rna_expression: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for key, tensor in rna_expression.items():
            rna_expression[key] = tensor.float().to(self.device)
        return rna_expression

    def compute_loss(
        self, survival_time: torch.Tensor, predictions: torch.Tensor
    ) -> Dict:
        survival_time = survival_time.reshape(-1)
        predictions = predictions.reshape(-1)
        loss_l1 = (survival_time - predictions).abs().mean()
        loss_l2 = (survival_time - predictions).pow(2).mean()
        accuracy_3_months = ((survival_time - predictions) < 0.25).sum() / len(
            predictions
        )
        accuracy_6_months = ((survival_time - predictions) < 0.50).sum() / len(
            predictions
        )
        accuracy_12_months = ((survival_time - predictions) < 1.00).sum() / len(
            predictions
        )
        loss = self.coeff_loss * loss_l2
        loss_dict = {
            self.loss_key: loss,
            self.loss_l1_key: loss_l1,
            self.loss_l2_key: loss_l2,
            self.accuracy_3_months_str: accuracy_3_months,
            self.accuracy_6_months_str: accuracy_6_months,
            self.accuracy_12_months_str: accuracy_12_months,
        }

        return loss_dict

    def save_logs(self, loss_dict: Dict) -> None:
        if (
            self.global_step % self.log_every_n_steps == 0
            and dst_tools.is_main_process()
        ):
            learning_rate = torch.tensor(self.optimizer.param_groups[0]["lr"])
            mlflow_logging.log_metric(
                key="Learning rate", value=learning_rate, step=self.global_step
            )
            mlflow_logging.log_metrics(loss_dict, self.global_step, prefix="Train")

    def training_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        rna_input, _, survival_time, _ = batch
        rna_input = self.data_to_device(rna_input)
        survival_time = self.target_to_device(survival_time)
        predictions = model(rna_input)
        loss_dict = self.compute_loss(survival_time, predictions)
        loss = loss_dict[self.loss_key]
        self.save_logs(loss_dict)

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return loss_dict

    def validation_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        rna_input, _, survival_time, _ = batch
        rna_input = self.data_to_device(rna_input)
        survival_time = self.target_to_device(survival_time)
        predictions = model(rna_input)
        loss_dict = self.compute_loss(survival_time, predictions)

        return loss_dict

    def test_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        return self.validation_step(model, batch, step)

    def test_step_onco(
        self, model: torch.nn.Module, batch: Tuple, step: int
    ) -> torch.Tensor:
        rna_input, rna_id = batch
        rna_input = self.data_to_device(rna_input)
        predictions = model(rna_input)
        predictions = predictions.reshape(-1)

        return predictions, rna_id

    def test_epoch_end(self, outputs: List) -> None:
        if dst_tools.is_main_process():
            mlflow_logging.log_metrics_list(
                outputs, self.global_step, prefix="TestEpoch"
            )

    def test_onco_epoch_end(self, outputs: List) -> None:
        prediction_list = []
        id_list = []
        for output in outputs:
            prediction, rna_id = output
            id_list = id_list + list(rna_id)
            prediction_list.append(prediction)

        predictions = torch.cat(prediction_list)
        sorted_indices = torch.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]
        sorted_ids = [id_list[int(index)] for index in sorted_indices]

        predictions_dict = {
            "sorted_ids": sorted_ids,
            "sorted_predictions": sorted_predictions,
        }
        artifacts_dir = os.path.join(os.path.dirname(self.save_directory), "artifacts")
        save_path = os.path.join(
            artifacts_dir, "onco_predictions_epoch_{}.pickle".format(self.epoch)
        )
        self.save_predictions(predictions_dict, save_path)

    def print_eval_status(self, prefix: str, step: int, n_batches: int) -> None:
        print("{} Epoch: {}, Step: {}/{}".format(prefix, self.epoch, step, n_batches))

    def save_predictions(self, predictions: Dict, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(predictions, handle)
        print("Onco predictions saved in: {}".format(path))
        return None

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        test_loader: data.DataLoader,
        onco_loader: data.DataLoader,
    ) -> None:
        n_training_batches, n_validation_batches = self.compute_n_batches(
            train_loader, val_loader
        )
        n_test_batches, n_onco_batches = self.compute_n_batches(
            test_loader, onco_loader
        )
        model, start_epoch = self.resume_training_if_provided(model)

        if self.dist_params["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.dist_params["gpu"]]
            )

        self.optimizer.zero_grad()
        for epoch_id in range(start_epoch, self.end_epoch):
            # Setting epoch id and loss accumulators
            self.epoch = epoch_id
            train_outputs, val_outputs = [], []
            test_outputs, onco_outputs = [], []

            # Training the model
            model.train()
            if self.dist_params["distributed"]:
                train_loader.sampler.set_epoch(epoch_id)
            for step, batch in enumerate(train_loader):
                loss_dict = self.training_step(model, batch, step)
                train_outputs.append(loss_dict)
                self.print_status("Train", step, n_training_batches, loss_dict)
            self.training_epoch_end(train_outputs)

            # Validation loop
            if val_loader is not None:
                torch.set_grad_enabled(False)
                model.eval()
                for step, batch in enumerate(val_loader):
                    loss_dict_val = self.validation_step(model, batch, step)
                    val_outputs.append(loss_dict_val)
                    self.print_status(
                        "Validation", step, n_validation_batches, loss_dict_val
                    )
                self.validation_epoch_end(val_outputs)
                torch.set_grad_enabled(True)

            # Evaluation loop
            if test_loader is not None:
                torch.set_grad_enabled(False)
                model.eval()
                for step, batch in enumerate(test_loader):
                    loss_dict_test = self.test_step(model, batch, step)
                    test_outputs.append(loss_dict_test)
                    self.print_status("Test", step, n_test_batches, loss_dict_test)
                self.test_epoch_end(test_outputs)
                torch.set_grad_enabled(True)

            # Evaluation loop on oncopole data
            if onco_loader is not None:
                torch.set_grad_enabled(False)
                model.eval()
                for step, batch in enumerate(onco_loader):
                    onco_predictions = self.test_step_onco(model, batch, step)
                    onco_outputs.append(onco_predictions)
                    self.print_eval_status("Test", step, n_onco_batches)
                self.test_onco_epoch_end(onco_outputs)
                torch.set_grad_enabled(True)

            self.scheduler.step()
            self.save_model(model, loss_dict)

            if self.dist_params["distributed"]:
                dist.barrier()


def get_att_trainer_survival_time(
    logging_params: Dict,
    trainer_params: Dict,
    save_directory: str,
    dist_params: Dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> torch.nn.Module:
    att_trainer = AttentionTrainerSurvivalTime(
        **logging_params,
        **trainer_params,
        save_directory=save_directory,
        dist_params=dist_params,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return att_trainer
