import numpy as np
from scipy.linalg import hadamard as H


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

    return fltr / (np.ceil(np.log2(N)) * dim)


def laplacian():
    return (
        np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ]
        )
        / 8
    )


def gaussian_blur():
    return (
        np.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )
        / 16
    )
