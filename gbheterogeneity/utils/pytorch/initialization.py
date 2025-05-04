import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def set_deterministic_start(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
