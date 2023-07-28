from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from itertools import chain, pairwise

import torch
from torch import cuda

if TYPE_CHECKING:
    from typing import Callable, Iterable, Any
    from numbers import Number
    from torch import Tensor

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]

log = logging.getLogger(__name__)
USE_CUDA = cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

if USE_CUDA:
    log.info(f"Using CUDA on device {torch.device('cuda')}")


def create_tensor(fn: type[Tensor] | Callable, /, *args, **kwargs):
    tensor = fn(*args, **kwargs)
    return tensor.to("cuda") if USE_CUDA else tensor


def init_params(size):
    return torch.nn.Parameter(torch.empty(size, requires_grad=True))


def reset_parameter(tensor):
    torch.nn.init.uniform_(tensor, 0, 2 * torch.pi)


def cut(
    arr: Iterable[Number], i: Number | Iterable[Number]
) -> Iterable[Iterable[Number]]:
    """
    Splits sequence at indice(s) i

    Args:
        arr (Iterable[Number]): Iterable to split
        i (Number | Iterable[Number]): indice(s) to divide by

    Returns:
        Tuple[Iterable[Number], Iterable[Number]]: split sub-sequences
    """
    if not is_iterable(i):
        i = (i,)

    index = chain((None,), i, (None,))
    return tuple(arr[a:b] for a, b, in pairwise(index))
