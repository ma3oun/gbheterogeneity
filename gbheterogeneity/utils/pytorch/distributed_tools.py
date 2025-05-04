import os
import torch

import torch.distributed as dist

from typing import Dict


RANK_LABEL = "RANK"
WORLD_SIZE_LABEL = "WORLD_SIZE"
SLURM_PROCID_LABEL = "SLURM_PROCID"
LOCAL_RANK_LABEL = "LOCAL_RANK"
NCCL_LABEL = "nccl"
APEX_LABEL = "USE_APEX"

DEFAULT_WORLD_SIZE = 0
DEFAULT_RANK = 0


def distributed_is_available_and_initialized() -> None:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not distributed_is_available_and_initialized():
        return DEFAULT_WORLD_SIZE
    return dist.get_world_size()


def get_rank() -> int:
    if not distributed_is_available_and_initialized():
        return DEFAULT_RANK
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode() -> Dict:
    dist_params = {}
    dist_params["url"] = "env://"

    if RANK_LABEL in os.environ and WORLD_SIZE_LABEL in os.environ:
        dist_params["rank"] = int(os.environ[RANK_LABEL])
        dist_params["world_size"] = int(os.environ[WORLD_SIZE_LABEL])
        dist_params["gpu"] = int(os.environ[LOCAL_RANK_LABEL])
    elif SLURM_PROCID_LABEL in os.environ:
        dist_params["rank"] = int(os.environ[SLURM_PROCID_LABEL])
        dist_params["gpu"] = dist_params["rank"] % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        dist_params["distributed"] = False
        dist_params["apex"] = False
        return dist_params

    dist_params["distributed"] = True
    if APEX_LABEL in os.environ:
        dist_params["apex"] = bool(os.environ[APEX_LABEL])
    else:
        dist_params["apex"] = False
    torch.cuda.set_device(dist_params["gpu"])
    dist_params["dist_backend"] = NCCL_LABEL

    print(
        "| Distributed init (rank {}): {}".format(
            dist_params["rank"], dist_params["url"]
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=dist_params["dist_backend"],
        init_method=dist_params["url"],
        world_size=dist_params["world_size"],
        rank=dist_params["rank"],
    )
    torch.distributed.barrier()
    setup_for_distributed(dist_params["rank"] == 0)
    return dist_params
