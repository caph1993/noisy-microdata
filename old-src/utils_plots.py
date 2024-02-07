import matplotlib.pyplot as plt
from itertools import product

import numpy as np
import time


class Timer:
    def __init__(self, label="") -> None:
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        human_readable = self.interval
        if human_readable > 60:
            human_readable /= 60
            unit = "minutes"
        else:
            unit = "seconds"
        s = f"{self.label}: " if self.label else ""
        print(f"{s}Elapsed time: {human_readable:.2f} {unit}", flush=True)


def plt_color(ax=None, color=None):
    if color is None:
        ax = ax or plt.gca()
        color = next(ax._get_lines.prop_cycler)["color"]
    return color


def plt_subplots_matrix(d, ax_mat=None, fig=None):
    if ax_mat is not None:
        return ax_mat
    if fig is None:
        fig = plt.gcf()
    ax_mat = fig.subplots(d, d)
    for r, c in product(range(d), repeat=2):
        ax: plt.Axes = ax_mat[r][c]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if r < d - 1:
            ax.set_xticks([])
        if (c > 0 and r < d - 1) or (c != 1 and r == d - 1):
            ax.set_yticks([])
    for i, j, ax, _ in plt_ij_iter(d, ax_mat):
        if j == 0:
            ax.set_xlabel(f"Column {i}")
        if i == 0:
            ax.set_ylabel(f"Column {j}")
    return ax_mat


class KwSplitter:
    def __init__(self):
        self._list = []

    def add_prefix(self, prefix, **defaults):
        self._list.append((prefix, {**defaults}))

    def split(self, kw):
        d = {k: {**v} for k, v in self._list}
        for k, v in kw.items():
            try:
                prefix = next(s for s, _ in self._list if k.startswith(s))
            except StopIteration:
                for prefix, _ in self._list:
                    d[prefix][k] = v
            else:
                k = k[len(prefix) :]
                d[prefix][k] = v
        return tuple(d[pre] for pre, _ in self._list)


def _kw_color_copy(kw, ax=None):
    kw = {**kw}
    if "color" not in kw:
        ax = ax or plt.gca()
        color = next(ax._get_lines.prop_cycler)["color"]
        kw["color"] = color
    return kw


def plt_ij_iter(d, ax_mat=None, kw_ii={}, kw_ij={}):
    "keywords starting with ii_ or ij_ are passed to diagonal or non-diagonal separately"
    ax_mat = plt_subplots_matrix(d, ax_mat)

    for i, j in product(range(d), repeat=2):
        ax: plt.Axes = ax_mat[d - 1 - j][i]
        kw = _kw_color_copy(kw_ii if i == j else kw_ij, ax)
        yield (i, j, ax, kw)
    return ax_mat


def kw_split_ij(kw_ii_ij, ii_defaults={}, ij_defaults={}):
    kw_ii = {**ii_defaults}
    kw_ij = {**ij_defaults}
    for k, v in kw_ii_ij.items():
        if k.startswith("ii_"):
            kw_ii[k[3:]] = v
        elif k.startswith("ij_"):
            kw_ij[k[3:]] = v
        else:
            kw_ii[k] = kw_ij[k] = v
    return kw_ii, kw_ij


def plt_mat_samples(samples, ax_mat=None, **kw_ii_ij):
    "keywords starting with hist_ or sc_ are passed to hist or scatter separately"
    defaults = dict(
        ii_density=True, ii_bins="sqrt", ii_alpha=0.6, ij_alpha=0.3, ij_marker="."
    )
    n, d = samples.shape
    ax_mat = plt_subplots_matrix(d, ax_mat)
    kw_ii, kw_ij = kw_split_ij(kw_ii_ij, *kw_split_ij(defaults))

    for i, j, ax, kw in plt_ij_iter(d, ax_mat, kw_ii, kw_ij):
        if i == j:
            ax.hist(samples[:, i], **kw)
        else:
            ax.scatter(samples[:, i], samples[:, j], **kw)
    return ax_mat


def discrete_marginals_plot(pX, ax_mat=None, fig=None, **kw):
    d = len(pX.shape)
    if fig is None:
        fig = plt.gcf()
    ax_mat = plt_subplots_matrix(d, ax_mat, fig)
    title = kw.pop("title", None)
    if title:
        fig.suptitle(title)

    defaults = dict(
        ij_interpolation="none",
        ij_origin="lower",
        ij_aspect="auto",
        ij_vmin=0,
        ii_alpha=0.6,
    )
    kw_ii, kw_ij = kw_split_ij(kw, *kw_split_ij(defaults))

    for i, j, ax, kw in plt_ij_iter(d, ax_mat, kw_ii, kw_ij):
        aux = pX.copy()
        for k in range(d):
            if k != i and k != j:
                aux = np.sum(aux, axis=k, keepdims=True)
        aux = np.squeeze(aux)
        if i > j:
            aux = aux.T
        if i == j:
            ax.bar(np.arange(len(aux)), aux, **kw_ii)  # type: ignore
        else:
            assert len(aux.shape) == 2
            extent = [-0.5, pX.shape[i] - 0.5, -0.5, pX.shape[j] - 0.5]
            ax.imshow(aux.T, extent=extent, **kw_ij)
    return ax_mat
