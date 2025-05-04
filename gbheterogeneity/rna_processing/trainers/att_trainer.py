import math
import torch

import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist

import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools
import gbheterogeneity.utils.pytorch.simple_distributed_trainer as simple_trainer

from typing import Dict, Tuple


class AttentionTrainer(simple_trainer.SimpleDistributedTrainer):

    def __init__(self,
                 start_epoch: int = 0,
                 max_epochs: int = 50,
                 temp: float = 0.005,
                 mask_prob: float = 0.30,
                 coeff_reconstruction_loss: float = 1.0,
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
                 **kwargs) -> None:
                
        super().__init__(start_epoch=start_epoch,
                         max_epochs=max_epochs,
                         device=device,
                         dist_params=dist_params,
                         log_every_n_steps=log_every_n_steps,
                         log_n_last_models=log_n_last_models,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         save_directory=save_directory,
                         resume_file=resume_file,
                         optimizer=optimizer,
                         scheduler=scheduler)

        self.warmup_epochs = warmup_epochs

        self.temp = temp
        self.mask_prob = mask_prob
        self.coeff_reconstruction_loss = coeff_reconstruction_loss
        self.coeff_contrastive_loss = coeff_contrastive_loss
        self.loss_reconstruction_key = "loss_reconstruction"
        self.loss_contrastive_key = "loss_contrastive"

    def duplicate_and_mask_tensor(self, rna_input: torch.Tensor) -> torch.Tensor:
        rna_input = rna_input.float()
        rna_input_a = rna_input.clone()
        rna_input_b = rna_input.clone()
        probability_matrix = torch.full(rna_input.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        rna_input_b[masked_indices] = 0.0
        rna_output = torch.cat([rna_input_a, rna_input_b])
        return rna_output

    def duplicated_and_mask(self, rna_expression: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rna_target ={}
        for key, tensor in rna_expression.items():
            rna_expression[key] = self.duplicate_and_mask_tensor(tensor)
            target = tensor.clone().float()
            rna_target[key] = torch.cat([target, target])
        return rna_expression, rna_target

    def data_to_device(self, rna_expression: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, tensor in rna_expression.items():
            rna_expression[key] = tensor.to(self.device)
        return rna_expression

    def compute_reconstruction_loss(self, rna_input: torch.Tensor, rna_reconstruction: torch.Tensor) -> torch.Tensor:
        assert len(rna_input) == len(rna_reconstruction)

        loss = 0.0
        for (key_in, tensor_in), (key_out, tensor_out) in zip(rna_input.items(), rna_reconstruction.items()):
            loss = loss + (tensor_in - tensor_out).pow(2).sum(dim=1).mean()
        loss = loss / len(rna_input)
        return loss

    def compute_contrastive_loss(self, rna_projections: torch.Tensor) -> torch.Tensor:
        batch_size, _ = rna_projections.shape
        batch_size = batch_size // 2
        rna_projections_a = rna_projections[:batch_size, :]
        rna_projections_b = rna_projections[batch_size:, :]
        similarity = rna_projections_a @ rna_projections_b.t() / self.temp
        targets = torch.zeros(similarity.size()).to(similarity.device)
        targets.fill_diagonal_(1)
        loss = -torch.sum(F.log_softmax(similarity, dim=1) * targets, dim=1).mean()
        return loss

    def compute_loss(self, 
                     rna_input: Dict[str, torch.Tensor], 
                     rna_reconstruction: Dict[str, torch.Tensor], 
                     rna_projections: torch.Tensor) -> Dict:
        loss_reconstruction = self.compute_reconstruction_loss(rna_input, rna_reconstruction)
        loss_contrastive = self.compute_contrastive_loss(rna_projections)

        loss = 0.0
        if self.coeff_reconstruction_loss:
            loss = loss + self.coeff_reconstruction_loss * loss_reconstruction
        if self.coeff_contrastive_loss:
            loss = loss + self.coeff_contrastive_loss * loss_contrastive

        loss_dict = {self.loss_key: loss,
                     self.loss_reconstruction_key: loss_reconstruction,
                     self.loss_contrastive_key: loss_contrastive}

        return loss_dict

    def save_logs(self, loss_dict: Dict) -> None:
        if self.global_step % self.log_every_n_steps == 0 and dst_tools.is_main_process():
            learning_rate = torch.tensor(self.optimizer.param_groups[0]["lr"])
            mlflow_logging.log_metric(key="Learning rate", value=learning_rate, step=self.global_step)
            mlflow_logging.log_metrics(loss_dict, self.global_step, prefix="Train")

    def training_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        rna_input, _ = batch
        rna_input, rna_target = self.duplicated_and_mask(rna_input)        
        rna_input = self.data_to_device(rna_input)
        rna_target = self.data_to_device(rna_target)       
        rna_reconstruction, rna_projections = model(rna_input)
        loss_dict = self.compute_loss(rna_target, rna_reconstruction, rna_projections)
        loss = loss_dict[self.loss_key]
        self.save_logs(loss_dict)

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.epoch == 0 and step % self.step_size == 0 and step <= self.warmup_iterations:
            self.scheduler.step(step // self.step_size)

        self.global_step += 1

        return loss_dict

    def validation_step(self, model: torch.nn.Module, batch: Tuple, step: int) -> Dict:
        rna_input, _ = batch
        rna_input, rna_target = self.duplicated_and_mask(rna_input)
        rna_input = self.data_to_device(rna_input)
        rna_target = self.data_to_device(rna_target)
        rna_reconstruction, _, rna_projections = model(rna_input)
        loss_dict = self.compute_loss(rna_target, rna_reconstruction, rna_projections)

        return loss_dict

    def fit(self, model: torch.nn.Module, train_loader: data.DataLoader, val_loader: data.DataLoader = None) -> None:
        n_training_batches, n_validation_batches = self.compute_n_batches(train_loader, val_loader)
        self.step_size = math.floor(n_training_batches / self.warmup_epochs)
        self.warmup_iterations = self.warmup_epochs * self.step_size
        model, start_epoch = self.resume_training_if_provided(model)

        if self.dist_params["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.dist_params["gpu"]])

        self.optimizer.zero_grad()

        for epoch_id in range(start_epoch, self.end_epoch):
            # Setting epoch id and loss accumulators
            self.epoch = epoch_id
            train_outputs, val_outputs = [], []

            # Update scheduler
            if epoch_id > 0:
                self.scheduler.step(epoch_id + self.warmup_epochs)

            # Training the model
            model.train()
            if self.dist_params["distributed"]:
                train_loader.sampler.set_epoch(epoch_id)            
            for step, batch in enumerate(train_loader):
                loss_dict = self.training_step(model, batch, step)
                train_outputs.append(loss_dict)
                self.print_status("Train", step, n_training_batches, loss_dict)
            self.training_epoch_end(train_outputs)

            # Evaluating the model
            if val_loader is not None:
                torch.set_grad_enabled(False)
                model.eval()
                for step, batch in enumerate(val_loader):
                    loss_dict_val = self.validation_step(model, batch, step)
                    val_outputs.append(loss_dict_val)
                    self.print_status("Validation", step, n_validation_batches, loss_dict_val)
                self.validation_epoch_end(val_outputs)
                torch.set_grad_enabled(True)

            self.save_model(model, loss_dict)

            if self.dist_params["distributed"]:
                dist.barrier()

def get_att_trainer(logging_params: Dict,
                    trainer_params: Dict,
                    save_directory: str,
                    dist_params: Dict,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler) -> torch.nn.Module:

    att_trainer = AttentionTrainer(**logging_params,
                                       **trainer_params,
                                       save_directory=save_directory,
                                       dist_params=dist_params,
                                       optimizer=optimizer,
                                       scheduler=scheduler)
    return att_trainer