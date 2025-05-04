import torch

from typing import List, Union, Tuple

def add_weight_decay(model: torch.nn.Module, weight_decay: float = 1e-5, skip_list: Tuple = ()) -> List:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]

def get_adam_optimizer(module_params: Union[List, torch.nn.Module], learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    if isinstance(module_params, list):
        optimizer = torch.optim.Adam(module_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(module_params.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer

def get_sgd_optimizer(model: torch.nn.Module, learning_rate: float, momentum:float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    return optimizer

def get_adamw_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    if weight_decay:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(parameters, **opt_args)

    return optimizer