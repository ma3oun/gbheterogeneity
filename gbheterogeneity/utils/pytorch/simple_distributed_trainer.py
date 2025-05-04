import os
import torch

import torch.utils.data as data
import torch.distributed as dist

import gbheterogeneity.utils.mlflow.pytorch_logging as mlflow_logging
import gbheterogeneity.utils.pytorch.generic_trainer as gen_trainer
import gbheterogeneity.utils.pytorch.distributed_tools as dst_tools

from typing import Union, List, Dict, Tuple
from tqdm import tqdm
#from apex import amp as ampp


class SimpleDistributedTrainer(gen_trainer.GenericTrainer):
    """Create a distributed training framework"""

    def __init__(
        self,
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
    ) -> None:

        super().__init__()
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(max_epochs)
        self.device = device
        self.dist_params = dist_params
        self.log_every_n_steps = log_every_n_steps
        self.log_n_last_models = log_n_last_models
        self.model_path_list = []
        self.checkpoint_path_list = []
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_directory = save_directory
        self.resume_file = resume_file
        self.global_step = 0
        self.epoch = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_key = "loss"

    def training_step(
        self, model: torch.nn.Module, batch: Union[List, Dict, Tuple]
    ) -> Dict:
        raise NotImplementedError

    def validation_step(
        self, model: torch.nn.Module, batch: Union[List, Dict, Tuple]
    ) -> Dict:
        raise NotImplementedError

    def data_to_device(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(
        self, model: torch.nn.Module, outputs: Union[List, Dict, Tuple]
    ) -> Dict:
        raise NotImplementedError

    def save_logs(self, image_batch: torch.Tensor, loss_dict: Dict) -> None:
        if (
            self.global_step % self.log_every_n_steps == 0
            and dst_tools.is_main_process()
        ):
            learning_rate = torch.tensor(self.scheduler.get_last_lr()[0])
            mlflow_logging.log_metric(
                key="Learning rate", value=learning_rate, step=self.global_step
            )
            mlflow_logging.log_metrics(loss_dict, self.global_step, prefix="Train")
            filename = "image_batch_{}.png".format(self.global_step)
            batch_size = image_batch.shape[0] // 2
            images = torch.cat(
                [image_batch[0:10], image_batch[batch_size : batch_size + 10]], dim=0
            )
            images = torch.clip(images, min=0.0, max=1.0)
            mlflow_logging.log_image_batch(
                images, filename, n_images=20, n_rows=5, prefix="Train"
            )

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        if dst_tools.is_main_process():
            mlflow_logging.log_metrics_list(
                outputs, self.global_step, prefix="TrainEpoch"
            )

    def validation_epoch_end(self, outputs: List) -> None:
        if dst_tools.is_main_process():
            mlflow_logging.log_metrics_list(
                outputs, self.global_step, prefix="ValEpoch"
            )

    def print_status(
        self,
        prefix: str,
        step: int,
        n_batches: int,
        loss_dict: Dict,
        tqdmBar: tqdm = None,
    ) -> None:
        if dst_tools.is_main_process():
            loss = loss_dict[self.loss_key]
            status = "{} Epoch: {}, Step: {}/{}, Loss: {}".format(
                prefix, self.epoch, step, n_batches, loss
            )
            if tqdmBar is None:
                print(status)
            else:
                tqdmBar.set_description(status)

    def compute_n_batches(
        self, train_loader: data.DataLoader, val_loader: data.DataLoader = None
    ) -> Tuple:
        n_training_batches = len(train_loader)
        n_validation_batches = None
        if val_loader is not None:
            n_validation_batches = len(val_loader)

        return n_training_batches, n_validation_batches

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader = None,
    ) -> None:
        n_training_batches, n_validation_batches = self.compute_n_batches(
            train_loader, val_loader
        )
        model, start_epoch = self.resume_training_if_provided(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.dist_params["gpu"]]
        )  # , find_unused_parameters=True)
        self.optimizer.zero_grad()

        for epoch_id in range(start_epoch, self.end_epoch):
            # Setting epoch id and loss accumulators
            self.epoch = epoch_id
            train_outputs, val_outputs = [], []

            # Training the model
            model.train()
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
                    self.print_status(
                        "Validation", step, n_validation_batches, loss_dict_val
                    )
                self.validation_epoch_end(val_outputs)
                torch.set_grad_enabled(True)

            self.scheduler.step()
            self.save_model(model, loss_dict)

            dist.barrier()

    def keep_n_checkpoints(self):
        if len(self.checkpoint_path_list) > self.log_n_last_models:
            self.remove_oldest_checkpoint()

    def remove_oldest_checkpoint(self):
        checkpoint_file = self.checkpoint_path_list.pop(0)
        model_file = self.model_path_list.pop(0)
        assert len(self.checkpoint_path_list) == self.log_n_last_models
        assert len(self.model_path_list) == self.log_n_last_models
        os.remove(checkpoint_file)
        os.remove(model_file)

    def save_model(
        self,
        model: torch.nn.Module,
        loss_dict: Dict[str, torch.Tensor],
        isBest: bool = False,
    ) -> None:
        if dst_tools.is_main_process():
            model_to_save = model.module if hasattr(model, "module") else model
            model_path = "trained_model_epoch_" + str(self.epoch) + ".bin"
            model_path = os.path.join(self.save_directory, model_path)
            torch.save(model_to_save.state_dict(), model_path)

            if isBest:
                print("***** Updating best model *****")
                new_best_model_path = os.path.join(
                    self.save_directory, f"best_trained_model_{self.epoch}.bin"
                )
                torch.save(model_to_save.state_dict(), new_best_model_path)
                if not self.best_model_path is None:
                    os.remove(self.best_model_path)
                self.best_model_path = new_best_model_path
            else:
                if not self.best_model_path is None:
                    print("----- Last validation was not the best -----")

            checkpoint_path = "ckpt_epoch" + str(self.epoch) + ".tar"
            checkpoint_path = os.path.join(self.save_directory, checkpoint_path)
            ckptData = {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "loss": loss_dict[self.loss_key].item(),
            }
            #if self.dist_params["apex"]:
            #    ckptData.update({"amp": ampp.state_dict()})
            torch.save(
                ckptData,
                checkpoint_path,
            )
            self.model_path_list.append(model_path)
            self.checkpoint_path_list.append(checkpoint_path)
            self.keep_n_checkpoints()

    def change_model_checkpoint_names(self, checkpoint: Dict) -> Dict:
        new_checkpoint = {}
        model_dict = checkpoint["model_state_dict"]
        for attribute in model_dict:
            if attribute.startswith("module."):
                key = attribute.replace(old="module.", new="", max=1)
                new_checkpoint[key] = model_dict[attribute]
            else:
                new_checkpoint[attribute] = model_dict[attribute]
        return new_checkpoint

    def resume_training(self, model: torch.nn.Module) -> Tuple:
        checkpoint = torch.load(self.resume_file, map_location="cpu")
        model_checkpoint = self.change_model_checkpoint_names(checkpoint)

        model.load_state_dict(model_checkpoint)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        #if self.dist_params["apex"]:
        #    ampp.load_state_dict(checkpoint["amp"])
        del checkpoint

        return model, epoch, loss

    def move_optimizer_tensors_to_gpu(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()

    def resume_training_if_provided(self, model: torch.nn.Module) -> Tuple:
        if self.resume_file:
            model, epoch, loss = self.resume_training(model)
            self.move_optimizer_tensors_to_gpu()
            print("Resuming at epoch: {} (Loss: {})".format(epoch, loss))
            return model, epoch
        else:
            print("Starting from scratch")
            return model, self.start_epoch
