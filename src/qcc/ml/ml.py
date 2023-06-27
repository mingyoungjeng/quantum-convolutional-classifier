from __future__ import annotations
from typing import TYPE_CHECKING

from math import pi
from itertools import chain, pairwise
from functools import partial, wraps
from abc import ABCMeta

import torch
from torch import cuda
from torch.nn import ParameterDict, Module as TorchModule

if TYPE_CHECKING:
    from typing import Callable, Iterable, Any
    from numbers import Number
    from torch import Tensor

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]
    MLFunction = Callable[[Iterable[Number], Iterable[Number]], Iterable[Number]]

USE_CUDA = cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def parameter(arg):
    def decorator(f, key=None):
        f.__parameter__ = key
        return f

    # arg is a function, key=None
    if callable(arg):
        return decorator(arg)

    # arg is a key. Generate decorator(key=arg)
    return wraps(arg)(partial(decorator, key=arg))


class ModuleMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, /, **kwargs):
        bases = *bases, TorchModule

        # if "parameter" not in namespace:
        #     namespace["parameter"] = parameter

        if "__init__" in namespace:
            old_init = namespace["__init__"]

            def pre_init(self, *args, **kwargs):
                TorchModule.__init__(self)
                old_init(self, *args, **kwargs)

            namespace["__init__"] = pre_init

        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __init__(mcls, *args, **kwargs):
        mcls._id = 0
        super().__init__(*args, **kwargs)

    def count(mcls, val) -> int:
        if val is None:
            val = mcls._id
            mcls._id += 1
        return str(val)

    def __call__(mcls, *args, **kwds):
        self = super().__call__(*args, **kwds)

        params = (getattr(self, attr) for attr in dir(self))
        params = (f for f in params if hasattr(f, "__parameter__"))
        params = ((f.__parameter__, init_params(f(), angle=True)) for f in params)
        params = [(mcls.count(k), v) for k, v in params]
        params.sort()

        self.__parameters__ = ParameterDict(params)
        return self


class Module(metaclass=ModuleMeta):
    parameter = parameter


def is_iterable(x: Any):
    return hasattr(x, "__iter__")


def create_tensor(fn: type[Tensor] | Callable, /, *args, **kwargs):
    tensor = fn(*args, **kwargs)
    return tensor.cuda() if USE_CUDA else tensor


def init_params(size, angle=False):
    params = torch.nn.Parameter(create_tensor(torch.randn, size, requires_grad=True))

    if angle:
        with torch.no_grad():
            params *= 2 * pi
    return params


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
