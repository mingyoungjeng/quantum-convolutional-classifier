from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import zip_longest
import numpy as np

if TYPE_CHECKING:
    from typing import Optional
    from numbers import Number


# TODO: padding, dilation
def convolution(
    img_data,
    fltr,
    stride=1,
    bounds: Optional[tuple[Number, Number]] = None,
):
    img = np.zeros_like(img_data)
    for idx, _ in np.ndenumerate(img):
        # idx = stride*np.array(idx)
        # idy = np.sum(np.divmod(idx, image_data.shape), axis=0)
        # if any(idy >= image_data.shape):
        #     continue

        # Create the window
        window = img_data
        for i, (id, f) in enumerate(zip_longest(idx, fltr.shape, fillvalue=1)):
            id *= stride
            window = window.take(range(id, id + f), mode="wrap", axis=i)

        # Covers for if fltr and image are different dimensionality (color images)
        window = window.reshape(fltr.shape, order="F")

        # Performs convolution operation
        img[tuple(idx)] = np.sum(window * fltr)

    if bounds is not None:
        low, high = bounds
        img = np.maximum(img, low * np.ones_like(img))
        img = np.minimum(img, high * np.ones_like(img))

    return np.abs(img)


def avg_filter(N: int, dim: int = 1):
    return np.ones([N for _ in range(dim)]) / (N**dim)


def sobel_filter(N, dim: int = 2, axis: int = 0):
    fltr = np.zeros([N for _ in range(dim)])

    middle = N // 2
    middle = [middle - 1, middle] if N % 2 == 0 else [middle, middle]

    vec = np.ones(N)
    vec[middle] += 1

    lim = (N - 1) // 2
    for i in 1 + np.arange(lim):
        for j, m in enumerate(middle):
            c = (-1) ** j * i

            idx = np.index_exp[:] * axis + np.index_exp[m - c]
            fltr[idx] = c * vec

    # normalizing factor is just the sum of all of the (abs() of all) components
    return fltr / np.sum(np.absolute(fltr))


def normal(x, sigma: int = 1):
    return np.exp(-((x / sigma) ** 2) / 2) / np.sqrt(2 * np.pi * sigma**2)


def gaussian_blur(N, sigma=1, dim: int = 2):
    centre = (N - 1) / 2
    fltr = np.zeros([N for _ in range(dim)])

    for idx in np.ndindex(fltr.shape):
        fltr[idx] = np.prod([normal(i - centre, sigma) for i in idx])

    fltr = fltr / np.sum(fltr)  # normalizing the matrix

    return fltr


def laplacian_of_gaussian(N, sigma=0.6, dim: int = 2):
    center = (N - 1) / 2
    fltr = gaussian_blur(N, sigma, dim)

    for idx in np.ndindex(fltr.shape):
        coeff = np.sum((i - center) ** 2 for i in idx) / (2 * sigma**2) - 1
        fltr[idx] = coeff * fltr[idx] * (np.sqrt(2 / sigma**2)) ** dim

    avg = np.sum(fltr) / np.prod(fltr.shape)
    for i in range(0, fltr.shape[0]):
        for j in range(0, fltr.shape[1]):
            fltr[i][j] = fltr[i][j] - avg  # we instead normalize the LoG kernel to 0

    return fltr
