import torch

import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools
import gbheterogeneity.utils.pytorch.simple_distributed_trainer as simple_trainer

from typing import Dict


class AETrainer(simple_trainer.SimpleDistributedTrainer):

    def __init__(self,
                 start_epoch: int = 0,
                 max_epochs: int = 50,
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

    def data_to_device(self, rna_expression: torch.Tensor) -> torch.Tensor:
        return rna_expression.to(self.device)

    def compute_loss(self, rna_input: torch.Tensor, rna_reconstruction: torch.Tensor) -> Dict:
        loss = (rna_input - rna_reconstruction).pow(2).sum(dim=1).mean()
        loss = {self.loss_key: loss}
        return loss

    def save_logs(self, loss_dict: Dict) -> None:
        if self.global_step % self.log_every_n_steps == 0 and dst_tools.is_main_process():
            learning_rate = torch.tensor(self.scheduler.get_last_lr()[0])
            mlflow_logging.log_metric(key="Learning rate", value=learning_rate, step=self.global_step)
            mlflow_logging.log_metrics(loss_dict, self.global_step, prefix="Train")

    def training_step(self, model: torch.nn.Module, rna_input: torch.Tensor, step: int) -> Dict:
        rna_input = self.data_to_device(rna_input)
        rna_reconstruction = model(rna_input)
        loss_dict = self.compute_loss(rna_input, rna_reconstruction)
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

    def validation_step(self, model: torch.nn.Module, rna_input: torch.Tensor, step: int) -> Dict:
        rna_input = self.data_to_device(rna_input)
        rna_reconstruction = model(rna_input)
        loss_dict = self.compute_loss(rna_input, rna_reconstruction)
        return loss_dict

def get_ae_trainer(logging_params: Dict,
                   trainer_params: Dict,
                   save_directory: str,
                   dist_params: Dict,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler) -> torch.nn.Module:

    ae_trainer = AETrainer(**logging_params,
                              **trainer_params,
                              save_directory=save_directory,
                              dist_params=dist_params,
                              optimizer=optimizer,
                              scheduler=scheduler)
    return ae_trainer