import torch

from typing import List


def get_gpu_device(id: int) -> torch.device:
    device = torch.device("cuda:" + str(id))
    return device


def get_cpu_device() -> torch.device:
    device = torch.device("cpu")
    return device


def get_number_of_available_gpus() -> int:
    n_gpus = torch.cuda.device_count()

    return n_gpus


def get_device(gpu_ids: List[int] = []) -> torch.device:
    n_required_gpus = len(gpu_ids)

    if n_required_gpus == 0:
        print("Running on cpu device")
        device = get_cpu_device()
        return device

    elif torch.cuda.is_available():
        print("Running on gpu device")
        n_gpus = get_number_of_available_gpus()

        if n_required_gpus > n_gpus:
            raise ValueError("Required gpus ({}) > available gpus ({})".format(n_required_gpus, n_gpus))

        elif n_required_gpus == 1:
            print("Running on single gpu: {}".format(gpu_ids[0]))
            device = get_gpu_device(id=gpu_ids[0])
            return device
        else:
            print("Running on multiple gpus: {}".format(gpu_ids))
            device = torch.device("cuda")
            return device
    else:
        raise ValueError("Cannot run on the selected devices: {}".format(gpu_ids))


def model_to_multiple_gpus(model: torch.nn.Module, device: torch.device, gpu_ids: List[int]) -> torch.nn.Module:
    print("Moving model to gpus: {}".format(gpu_ids))
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    return model


def model_to_single_gpu(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    print("Moving model to gpu: {}".format(device))
    model.to(device)

    return model


def model_to_cpu(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    print("Moving model to cpu: {}".format(device))
    model.to(device)

    return model


def model_to_device(model: torch.nn.Module, device: torch.device, gpu_ids: List[int]) -> torch.nn.Module:
    n_required_gpus = len(gpu_ids)

    if n_required_gpus == 0:
        model = model_to_cpu(model, device=device)
    elif n_required_gpus == 1:
        model = model_to_single_gpu(model, device=device)
    else:
        model = model_to_multiple_gpus(model, device=device, gpu_ids=gpu_ids)

    return model
