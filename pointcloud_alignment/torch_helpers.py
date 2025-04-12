import torch
import numpy as np

def maxloc(x: torch.Tensor | np.ndarray) -> tuple:
    d = x.argmax().item()
    res = []
    for s in x.shape[::-1]:
        d, m = divmod(d, s)
        res.append(m)
    return tuple(res)[::-1]
