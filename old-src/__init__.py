from typing import Dict
from .utils_dataset import load_human_height
from .mechanisms import *
from .utils_cda import pc, ges, lingam, PostCDA
from .utils_silence import SilenceStdout
from .csv_writer import CSV_writer
from . import utils_plots

NDFloat = np.ndarray
NDInt = np.ndarray


def sub_sup_sample(data, m) -> NDFloat:
    n = len(data)
    div, mod = divmod(m, n)
    out = []
    if div > 0:
        out.extend([data] * div)
    if mod > 0:
        out.append(data[np.random.choice(n, size=mod)])
    out = np.concatenate(out, axis=-2)
    assert len(out) == m
    return out


def runIBU(ch: DiscreteChannel, pY: np.ndarray, nIters=150):
    assert ch.out_shape == pY.shape
    pX = np.ones(ch.in_shape)
    pX /= pX.size
    A = ch.matrix.reshape((pX.size, pY.size))
    A_AT = A @ A.T  # reuse this expression for speed
    pX = pX.reshape(-1)
    pY = pY.reshape(-1)
    for _ in range(nIters):
        pX = pX * (A @ pY) / (A_AT @ pX)
        # Numerical stability: (in theory is not needed, but in practice it is)
        pX /= pX.sum()
    pX = pX.reshape(ch.in_shape)
    return pX


def channelInversion(ch: DiscreteChannel, pY: np.ndarray, fix_negatives=True):
    assert ch.out_shape == pY.shape
    A = ch.matrix.reshape(-1, pY.size)
    A = np.linalg.inv(A)
    pY = pY.reshape(-1)
    pX = A @ pY
    # The output might be outside of the simplex. Fix with the greedy rule:
    # FIXME: ask if there is a better method for this part
    if np.any(pX < 0) and fix_negatives:
        pX[pX < 0] = 0
    pX /= pX.sum()
    pX = pX.reshape(ch.in_shape)
    return pX


from itertools import product


def param_grid(param_settings, shuffle=True, center=None):
    param_grid = [
        dict(zip(param_settings.keys(), params))
        for params in product(*param_settings.values())
    ]

    if center is not None:
        center = {} if center is True else center
        center = {**{k: v[0] for k, v in param_settings.items()}, **center}
        hamming = lambda d1, d2: sum(1 for k in d1.keys() if d1[k] != d2[k])
        param_grid = [params for params in param_grid if hamming(params, center) <= 1]

    param_grid = np.array(param_grid, dtype=object)
    if shuffle:
        np.random.shuffle(param_grid)
    return param_grid


def make_bounds(data, over_bound):
    low = data.min(axis=0)
    up = data.max(axis=0)
    ranges = up - low
    up += ranges * over_bound
    low -= ranges * over_bound
    bounds = [*zip(low, up)]
    return bounds
