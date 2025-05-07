import torch

from .cosine_scheduler import CosineLRScheduler


def get_exponential_scheduler(
    optimizer: torch.optim.Optimizer, gamma: float
) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return scheduler


def get_step_scheduler(
    optimizer: torch.optim.Optimizer, step_size: int, gamma: float
) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    return scheduler


def get_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    min_lr: float,
    decay_rate: int,
    warmup_lr: float,
    warmup_epochs: int,
) -> CosineLRScheduler:
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        t_mul=1.0,
        lr_min=min_lr,
        decay_rate=decay_rate,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_epochs,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
    )

    return lr_scheduler
