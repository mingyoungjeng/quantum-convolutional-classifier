from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain, pairwise
from torch import cuda

if TYPE_CHECKING:
    from typing import Callable, Iterable, Any
    from numbers import Number
    from torch import Tensor

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]
    MLFunction = Callable[[Iterable[Number], Iterable[Number]], Iterable[Number]]

USE_CUDA = cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def is_iterable(x: Any):
    return hasattr(x, "__iter__")


def create_tensor(fn: type[Tensor] | Callable, /, *args, **kwargs):
    tensor = fn(*args, **kwargs)
    return tensor.cuda() if USE_CUDA else tensor


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
