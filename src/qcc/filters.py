"""
_summary_
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import zip_longest

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Sequence
    from numbers import Number


# TODO: padding, dilation
def convolution(
    img_data: np.array,
    kernel: np.array,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    bounds: Optional[tuple[Number, Number]] = None,
    use_abs: bool = True,
) -> np.array:
    """
    Direct implementation of convolution

    Args:
        img_data (np.array): Input (multidimensinal) data
        kernel (np.array): Input (multidimensinal) filter/kernel
        stride (int | Sequence[int]): Distance to shift each window. Defaults to 1.
        padding (int | Sequence[int]): NOT IMPLEMENTED. Defaults to 0.
        dilation (int | Sequence[int]): NOT IMPLEMENTED. Defaults to 1.
        bounds (Optional[tuple[Number, Number]]): (Min, Max) of data values. Defaults to None.
        use_abs (bool): Takes the magnitude of result - needed for quantum compatibility. Defaults to True.

    Returns:
        np.array: Output (multidimensinal) data
    """
    shape_out = update_dims(
        img_data.shape,
        padding=padding,
        dilation=dilation,
        kernel_size=kernel.shape,
        stride=stride,
    )
    img = np.zeros(shape_out)

    for idx, _ in np.ndenumerate(img):
        # Create the window
        window = img_data
        for i, (id, f) in enumerate(zip_longest(idx, kernel.shape, fillvalue=1)):
            id *= stride
            window = window.take(range(id, id + f), mode="wrap", axis=i)

        # Covers for if kernel and image are different dimensionality (color images)
        window = window.reshape(kernel.shape, order="F")

        # Performs kernel / MAC operation
        img[tuple(idx)] = np.sum(window * kernel)

    if bounds is not None:
        low, high = bounds
        img = np.maximum(img, low * np.ones_like(img))
        img = np.minimum(img, high * np.ones_like(img))

    return np.abs(img) if use_abs else img


def update_size(
    size: int,
    padding: int = 0,
    dilation: int = 1,
    kernel_size: int = 2,
    stride: int = 1,
) -> int:
    """
    Calculate output size after convolution/pooling for one dimension

    Args:
        size (int): Input data size
        padding (int): Defaults to 0.
        dilation (int): Defaults to 1.
        kernel_size (int): Defaults to 2.
        stride (int): Defaults to 1.

    Returns:
        int: Output data size
    """
    size += 2 * padding
    size += -dilation * (kernel_size - 1)
    size += -1
    size = size // stride
    size += 1

    return size


def update_dims(
    dims: int | Sequence[int],
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 1,
) -> int | Sequence[int]:
    """
    Calculate output size after multidimensional convolution/pooling.
    Integer-type arguments assume symmetry across dimensions.

    Args:
        dims (int | Sequence[int]): Input data size
        kernel_size (int | Sequence[int]): Defaults to 2.
        stride (int | Sequence[int]): Defaults to 1.
        padding (int | Sequence[int]): Defaults to 0.
        dilation (int | Sequence[int]): Defaults to 1.

    Returns:
        int | Sequence[int]: Output data size
    """
    if isinstance(kernel_size, int):  # Handle integer value for kernel_size
        kernel_size = [kernel_size] * len(dims)

    ### Handle integer values for other parameters
    params = {
        "size": dims,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }
    params = dict(  # Type checking
        (key, [value] * len(kernel_size)) if isinstance(value, int) else (key, value)
        for key, value in params.items()
    )

    ### All arguments need to be sequences of the same length
    # Fill shorter arguments with default values
    # padding = 0
    # dilation = 1
    # kernel_size = 1
    # stride = 1

    # zip_longest with different fill values
    longest = max(len(value) for value in params.values())
    for key, value in params.items():
        if len(value) == longest:
            continue

        match key:
            case "padding":
                fill_value = 0
            case _:
                fill_value = 1

        value_new = (value[i] if i < len(value) else fill_value for i in range(longest))
        params[key] = tuple(value_new)

    new_dims = tuple(update_size(**dict(zip(params, t))) for t in zip(*params.values()))
    # new_dims = tuple(dim for dim in new_dims if dim > 1)  # Squeeze
    return new_dims


def avg_filter(N: int, dim: int = 1) -> np.array:
    """
    _summary_

    Args:
        N (int): _description_
        dim (int, optional): _description_. Defaults to 1.

    Returns:
        np.array: _description_
    """
    return np.ones([N for _ in range(dim)]) / (N**dim)


def sobel_filter(N, dim: int = 2, axis: int = 0):
    """
    _summary_

    Args:
        N (_type_): _description_
        dim (int, optional): _description_. Defaults to 2.
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    kernel = np.zeros([N for _ in range(dim)])

    middle = N // 2
    middle = [middle - 1, middle] if N % 2 == 0 else [middle, middle]

    vec = np.ones(N)
    vec[middle] += 1

    lim = (N - 1) // 2
    for i in 1 + np.arange(lim):
        for j, m in enumerate(middle):
            c = (-1) ** j * i

            idx = np.index_exp[:] * axis + np.index_exp[m - c]
            kernel[idx] = c * vec

    # normalizing factor is just the sum of all of the (abs() of all) components
    return 2 * kernel / np.sum(np.abs(kernel))


def normal(x, sigma: int = 1):
    """
    Normal distribution.
    Used when computing Gaussian blur.

    Args:
        x (_type_): _description_
        sigma (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    return np.exp(-((x / sigma) ** 2) / 2) / np.sqrt(2 * np.pi * sigma**2)


def gaussian_blur(N, sigma=1, dim: int = 2):
    """
    _summary_

    Args:
        N (_type_): _description_
        sigma (int, optional): _description_. Defaults to 1.
        dim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    centre = (N - 1) / 2
    kernel = np.zeros([N for _ in range(dim)])

    for idx in np.ndindex(kernel.shape):
        kernel[idx] = np.prod(tuple(normal(i - centre, sigma) for i in idx))

    kernel = kernel / np.sum(kernel)  # normalizing the matrix

    return kernel


def laplacian_of_gaussian(N, sigma=0.6, dim: int = 2):
    """
    _summary_

    Args:
        N (_type_): _description_
        sigma (float, optional): _description_. Defaults to 0.6.
        dim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    center = (N - 1) / 2
    kernel = gaussian_blur(N, sigma, dim)

    for idx in np.ndindex(kernel.shape):
        coeff = np.sum((i - center) ** 2 for i in idx) / (2 * sigma**2) - 1
        kernel[idx] = coeff * kernel[idx] * (np.sqrt(2 / sigma**2)) ** dim

    avg = np.sum(kernel) / np.prod(kernel.shape)
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            kernel[i][j] = (
                kernel[i][j] - avg
            )  # we instead normalize the LoG kernel to 0

    return kernel


# TODO: only works for odd right now
def laplacian_approx(N, dim: int = 2):
    """
    Integer approximation of Laplacian outline

    Args:
        N (_type_): _description_
        dim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    centre = (N - 1) // 2
    kernel = np.ones([N for _ in range(dim)])

    idx = tuple(centre for _ in range(dim))
    kernel[idx] = -(N**dim - 1)

    norm = (N**dim) - N
    return kernel / norm
