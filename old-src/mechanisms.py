import abc
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from itertools import product
import warnings
import scipy.special


Shape = Tuple[int, ...]


# Shortcut for simpler notation:
def classical(
    mechanism_name: str,
    input_shape,
    p_max: float,
):
    assert mechanism_name in ["krr", "krrC", "geo", "geoC"], mechanism_name
    p_max_col_by_col = p_max ** (1 / len(input_shape))
    if mechanism_name == "krrC":
        M = nDkRR_Mechanism.from_p_max(input_shape, p_max=p_max)
    elif mechanism_name == "geoC":
        M = nDGeometricMechanism(input_shape, p_max=p_max)
    elif mechanism_name == "krr":
        M = HStackedMechanisms(
            *[
                K_RR_Mechanism.from_p_max(n_bins, p_max=p_max_col_by_col)
                for n_bins in input_shape
            ]
        )
    elif mechanism_name == "geo":
        M = HStackedMechanisms(
            *[
                Geometric1D_Mechanism(n_bins, p_max=p_max_col_by_col)
                for n_bins in input_shape
            ]
        )
    else:
        raise Exception(mechanism_name)
    return M


class BinsGrid(list):
    n_bins: Shape

    def __init__(self, grid):
        super().__init__(grid)
        self.d = len(grid)
        self.n_bins = tuple([len(b) - 1 for b in grid])
        self.shape = self.n_bins  # alias
        self.bounds = np.array([(b[0], b[-1]) for b in grid])

    @classmethod
    def uniform(cls, n_bins: Shape, bounds):
        bounds = np.asarray(bounds)
        d = len(n_bins)
        assert bounds.shape == (d, 2)
        grid = [
            np.linspace(left, right, n + 1) for (left, right), n in zip(bounds, n_bins)
        ]
        return cls(grid)

    def discretize(self, X: np.ndarray):
        assert len(X.T) == self.d
        Y = []
        for i in range(len(X.T)):
            bins_i = self[i]
            n_bins_i = self.n_bins[i]
            colY = np.digitize(X.T[i], bins=bins_i)
            colY = np.clip(colY, a_min=0, a_max=n_bins_i - 1)
            Y.append(colY)
        Y = np.stack(Y).T
        return Y

    def cell(self, Y: np.ndarray):
        assert np.issubdtype(Y.dtype, int)
        center = []
        radius = []
        for i, colY in enumerate(Y.T):
            left = self[i][colY]
            right = self[i][colY + 1]
            center_i = (right + left) / 2
            radius_i = (right - left) / 2
            center.append(center_i)
            radius.append(radius_i)
        center = np.stack(center).T
        radius = np.stack(radius).T
        return center, radius

    def random_cell_sample(self, Y: np.ndarray):
        center, radius = self.cell(Y)
        out_cont = np.random.random(Y.shape) * 2 - 1
        out_cont = out_cont * radius + center
        return out_cont

    def marginalsX_2D(self, pX):
        d = len(self.shape)
        _pX = pX
        out = {}
        for i, j in product(range(d), repeat=2):
            pX = _pX.copy()
            for k in range(d):
                if k != i and k != j:
                    pX = np.sum(pX, axis=k, keepdims=True)
            pX = np.squeeze(pX)
            if i > j:
                pX = pX.T
            if i == j:
                out[(i, j)] = pX
            else:
                assert len(pX.shape) == 2
                out[(i, j)] = pX
        return out


def pEmpirical(Y: np.ndarray, shape):
    out = np.zeros(shape)
    Y_unique, cnt = np.unique(Y, axis=0, return_counts=True)
    p_cnt = cnt / cnt.sum()
    out[tuple(Y_unique.T)] = p_cnt
    return out


class Mechanism(abc.ABC):
    d: int  # Number of dimensions for a single input
    input_dtype = float
    output_dtype = float

    @abc.abstractmethod
    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        """
        Output has shape broadcast(Y, givenX)[:-1]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X

    def make_channel(self) -> "DiscreteChannel":
        raise NotImplementedError

    def _parseX(self, X: np.ndarray, ndims: int):
        X = np.asarray(X)
        in_shape = X.shape
        while len(X.shape) > ndims and X.shape[-1] == 1:
            X = X.reshape(X.shape[:-1])
        assert (
            ndims - 1 <= len(X.shape) <= ndims
        ), f"Expected {ndims-1} or {ndims} dimensions, got {len(X.shape)}: {X.shape}"
        if len(X.shape) == ndims - 1:
            X = X[None, :]
        n = len(X)
        return X, n, in_shape

    def _check_dtypes(self, X=None, Y=None):
        if X is not None:
            assert np.issubdtype(
                X.dtype, self.input_dtype
            ), f"Expected input dtype {self.input_dtype}, got {X.dtype}"
        if Y is not None:
            assert np.issubdtype(
                Y.dtype, self.output_dtype
            ), f"Expected input dtype {self.output_dtype}, got {Y.dtype}"


class LaplaceMechanism(Mechanism):
    input_dtype = float
    output_dtype = float

    def __init__(self, d: int, b: float):
        self.d = d
        # arXiv:2206.03396 Laplace mechanism 2022
        Gamma = scipy.special.gamma
        eps = 1 / b
        num = eps**d * Gamma(d / 2)
        den = 2 * np.pi ** (d / 2) * Gamma(d)
        c = num / den
        self._aux = (d, b, eps, c)

    def __call__(self, X):
        X, n, in_shape = self._parseX(X, 2)
        d, b, *_ = self._aux
        radial = np.random.normal(size=(n, d))
        radial /= np.linalg.norm(radial, axis=-1, keepdims=True)
        radial *= np.random.exponential(b, n)[:, None]
        Y = X + radial
        return Y.reshape(in_shape)

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        _, b, _, c = self._aux
        dist = np.linalg.norm(givenX - Y, axis=-1)  # type:ignore
        return c * np.exp(-dist / b)


class K_RR_Mechanism(Mechanism):
    d = 1
    input_dtype = int
    output_dtype = int

    def __init__(self, k: int, q: float):
        """q is the prob. of sampling noise uniformly, including the input"""
        self._aux = k, q

    @classmethod
    def from_p_max(cls, k: int, p_max: float):
        q = (1 - p_max) / (1 - 1 / k)
        return cls(k, q)

    def __call__(self, X):
        X, n, in_shape = self._parseX(X, 1)
        k, q = self._aux
        pure_noise = np.random.choice(k, n)
        mask = np.random.random(n) < q
        Y = np.where(mask, pure_noise, X)
        return Y.reshape(in_shape)

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        k, q = self._aux
        mask = Y == givenX
        if len(mask.shape) > 1:
            assert mask.shape[-1] == 1
            mask = mask[..., 0]
        prob = mask * (1 - q) + (~mask) * q / k
        return prob

    def make_channel(self):
        k, q = self._aux
        mat = np.eye(k) * (1 - q) + np.ones((k, k)) * q / k
        return DiscreteChannel(mat, self.d)


class nDkRR_Mechanism(Mechanism):
    input_dtype = int
    output_dtype = int

    def __init__(self, shape: Shape, q: float):
        """q is the prob. of sampling noise uniformly, including the input"""
        k_total = np.product(shape).astype(int)
        self.d = len(shape)
        self._aux = shape, q, k_total

    @classmethod
    def from_p_max(cls, shape: Shape, p_max: float):
        k = np.product(shape).astype(int)
        if p_max < 1 / k:
            warnings.warn("Expected p_max >= 1/k. Non-sense. Clipping p_max to 1/k")
            p_max = 1 / k
        q = (1 - p_max) / (1 - 1 / k)
        return cls(shape, q)

    def __call__(self, X):
        X, n, in_shape = self._parseX(X, 2)
        shape, q, _ = self._aux
        pure_noise = np.stack([np.random.choice(k, n) for k in shape]).T
        mask = np.random.random(n) < q
        Y = np.where(mask[:, None], pure_noise, X)
        return Y.reshape(in_shape)

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        _, q, k_total = self._aux
        mask = np.all(Y == givenX, axis=-1)
        if len(mask.shape) > 1:
            assert mask.shape[-1] == 1
            mask = mask[..., 0]
        prob = mask * (1 - q) + (~mask) * q / k_total
        return prob

    def make_channel(self):
        shape, q, k_total = self._aux
        mat = np.eye(k_total) * (1 - q) + np.ones((k_total, k_total)) * q / k_total
        mat = mat.reshape((*shape, *shape))
        return DiscreteChannel(mat, self.d)


def binary_search_positive(func: Callable, target_y: float, resolution_x=1e-9):
    """
    Returns the first positive value x (up to resolution_x) such that func(x) >= target_y
    Assumes func is non_decreasing.
    """
    lo = 1
    hi = 1
    while func(lo) > target_y:
        lo /= 2
    while func(hi) < target_y:
        hi *= 2
    while hi - lo > resolution_x:
        mid = (lo + hi) / 2
        if func(mid) >= target_y:
            hi = mid
        else:
            lo = mid
    return hi


class Geometric1D_Mechanism(Mechanism):
    """
    Geometric mechanism parametrized with maximum prob.
    No cliping, no scaling. Instead, for each input x, the distribution of y has a different decay rate eps, so that all decays are exponential and within the bounds.

    Input domain: [0..k-1]
    Output domain: [-over_bound..k-1+over_bound]

    """

    d = 1
    input_dtype = int
    output_dtype = int

    def __init__(self, k: int, p_max: float, over_bound: int = 0):
        arr_x = np.arange(k)
        arr_y = np.arange(k + 2 * over_bound) - over_bound
        if p_max < 1 / k:
            warnings.warn("Expected p_max >= 1/k. Non-sense. Clipping p_max to 1/k")
            p_max = 1 / k

        def find_eps(x):
            f = lambda eps: 1 / np.sum(np.exp(-eps * np.abs(arr_y - x)))
            return binary_search_positive(f, p_max)

        eps = np.array([find_eps(x) for x in arr_x])
        channel = np.exp(-eps[:, None] * np.abs(arr_x[:, None] - arr_y))
        channel /= np.sum(channel, axis=-1, keepdims=True)
        assert np.allclose(channel.max(axis=-1), p_max), (channel.max(axis=-1), p_max)

        self.p_max = p_max
        self._aux = k, over_bound, channel

    def __call__(self, X):
        X, n, in_shape = self._parseX(X, 1)
        k, ob, channel = self._aux
        Y = np.zeros_like(X)
        if n > 2 * k:
            # Faster with k chunks based on X
            for x in range(k):
                mask = X == x
                m = mask.sum()
                Y[mask] = np.random.choice(k + 2 * ob, size=m, p=channel[x]) - ob
        else:
            # Faster one by one:
            for i in range(n):
                Y[i] = np.random.choice(k + 2 * ob, p=channel[X[i]]) - ob
        return Y.reshape(in_shape)

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        *_, channel = self._aux
        return channel[givenX, Y]

    def make_channel(self):
        *_, channel = self._aux
        return DiscreteChannel(channel, self.d)


class nDGeometricMechanism(Mechanism):
    """
    Geometric mechanism parametrized with maximum prob.
    No cliping, no scaling. Instead, for each input x, the distribution of y has a different decay rate eps, so that all decays are exponential and within the bounds.

    Input domain: [0..k-1]
    Output domain: [-over_bound..k-1+over_bound]
    """

    input_dtype = int
    output_dtype = int

    def __init__(
        self,
        input_shape: Shape,
        p_max: float,
        over_bound: int = 0,
        ord: Any = 2,
        memory_limit=10000000,
    ):
        self.d = len(input_shape)
        ob = over_bound
        output_shape = tuple(k + 2 * ob for k in input_shape)

        k = int(np.prod(input_shape))
        if p_max < 1 / k:
            warnings.warn("Expected p_max >= 1/k. Non-sense. Clipping p_max to 1/k")
            p_max = 1 / k

        X = np.array([*np.ndindex(input_shape)])
        Y = np.array([*np.ndindex(output_shape)]) - ob

        memory = int(np.prod([*input_shape, *output_shape]))
        assert (
            memory <= memory_limit
        ), f"Array of size {memory} violates the memory_limit {memory_limit}. Please increase it if you are sure with the command ThisMechanism(...params, memory_limit=...)"

        def find_eps(x):
            norms = np.linalg.norm(Y - x, axis=-1, ord=ord)
            d_dist, n_dist = np.unique(norms, return_counts=True)
            f = lambda eps: 1 / np.sum(n_dist * np.exp(-eps * d_dist))
            return binary_search_positive(f, p_max)

        eps = np.array([find_eps(x) for x in X])
        channel = np.exp(
            -eps[:, None]
            * np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1, ord=ord)
        )
        assert channel.shape == (len(X), len(Y))  # clarity
        channel /= np.sum(channel, axis=-1, keepdims=True)
        assert np.allclose(channel.max(axis=-1), p_max), (channel.max(axis=-1), p_max)
        channel = channel.reshape((*input_shape, *output_shape))

        self.p_max = p_max
        self._aux = input_shape, over_bound, Y, channel

    def __call__(self, X):
        X, n, in_shape = self._parseX(X, 2)
        input_shape, _, arr_Y, channel = self._aux
        channel = channel.reshape((*input_shape, -1))
        Y = np.zeros_like(X)
        for i in range(n):
            p = channel[tuple(X[i])]
            Y[i, :] = arr_Y[np.random.choice(len(p), p=p)]
        return Y.reshape(in_shape)

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        *_, channel = self._aux
        return channel[givenX, Y]

    def make_channel(self):
        *_, channel = self._aux
        return DiscreteChannel(channel, self.d)


class DiscreteChannel:
    matrix: np.ndarray
    d_input: int

    def __init__(self, matrix, d_input):
        self.matrix = matrix
        self.d = len(matrix.shape)
        self.d_input = d_input
        self.d_output = self.d - d_input
        self.in_shape = matrix.shape[:d_input]
        self.out_shape = matrix.shape[d_input:]

    @classmethod
    def hstack(cls, *channels: "DiscreteChannel"):
        in_shapes = [ch.in_shape for ch in channels]
        out_shapes = [ch.out_shape for ch in channels]
        in_shape = sum([sh for sh in in_shapes], start=())
        out_shape = sum([sh for sh in out_shapes], start=())
        A = np.ones(in_shape + out_shape)
        d_in = 0
        d_out = len(in_shape)
        for i, ch in enumerate(channels):
            d_in2 = d_in + len(in_shapes[i])
            d_out2 = d_out + len(out_shapes[i])
            axes = [*range(d_in, d_in2), *range(d_out, d_out2)]
            d_in, d_out = d_in2, d_out2
            # Swap axes to the end so that broadcast product works
            tail = [*range(-len(axes), 0)]
            A = np.moveaxis(A, source=axes, destination=tail)
            A *= ch.matrix
            A = np.moveaxis(A, source=tail, destination=axes)
        return cls(A, len(in_shape))


class DiscretizedMechanism(Mechanism):
    output_dtype = int

    def __init__(self, contMechanism: Mechanism, grid: BinsGrid):
        self.input_dtype = contMechanism.input_dtype
        self.d = contMechanism.d
        self.M = contMechanism
        self.grid = grid

    def __call__(self, X: np.ndarray):
        return self.grid.discretize(self.M(X))

    def p(self, Y: np.ndarray, givenX: np.ndarray, mY=None) -> np.ndarray:
        "Y is discrete, X is continuous"
        self._check_dtypes(X=givenX, Y=Y)
        # For each y in Y, get several points in the cell of y
        prob = []
        for _ in range(mY or 1):
            centers, radius = self.grid.cell(Y)
            volume = np.product(radius * 2, axis=-1)
            if mY is None:
                Yi = centers
            else:
                Yi = self.grid.random_cell_sample(Y)
            pdf = self.M.p(Yi, givenX=givenX)
            prob.append(pdf * volume)
        prob = np.stack(prob).T
        return np.mean(prob, axis=-1)

    def make_channel(self, gridX: BinsGrid, mX=None, mY=None, tqdm=None):
        """
        if mX is None (same applies to mY), the center of the cells are used.
        Otherwise, mX (mY) points are sampled uniformly from each of the cells.
        """
        gridY = self.grid
        # assert gridY is not None, "Expected output_grid attribute on mechanism"
        x_shape = gridX.n_bins
        y_shape = gridY.n_bins
        scales = []
        tqdm = tqdm or (lambda it: it)
        A = np.zeros((*x_shape, *y_shape))
        for ii in tqdm(np.ndindex(*x_shape)):
            for jj in np.ndindex(*y_shape):
                if mX is None:
                    centers, _ = gridY.cell(np.array([ii]))
                    points = centers
                else:
                    points = np.stack([ii] * mX)
                    points = gridX.random_cell_sample(points)
                probs = self.p(np.asarray(jj), givenX=points, mY=mY)
                A[(*ii, *jj)] = np.mean(probs)
            scale = A[ii].sum()
            A[ii] /= scale
            scales.append(scale)
        scales = np.asarray(scales)
        low, high = scales.min(), scales.max()
        if low < 1 - 0.1 or high > 1 + 0.1:
            print(
                f"Warning: data was rescaled to add up to 1, but sums varied from {low:.0e} to {high:.1f}, with average of {scales.mean():.3f}"
            )
        return DiscreteChannel(A, self.d)


class HStackedMechanisms(Mechanism):
    def __init__(self, *mechanisms):
        self.d = np.sum([m.d for m in mechanisms])
        self.input_dtype = np.result_type(*[m.input_dtype for m in mechanisms])
        self.output_dtype = np.result_type(*[m.output_dtype for m in mechanisms])

        self.mechanisms = mechanisms
        mk_filter = lambda d0, d1, dtype: lambda X: X[..., d0:d1].astype(dtype)
        cum_d = np.cumsum([0] + [m.d for m in mechanisms])
        zipX = []
        zipXY = []
        for i, M in enumerate(self.mechanisms):
            fX = mk_filter(cum_d[i], cum_d[i + 1], M.input_dtype)
            fY = mk_filter(cum_d[i], cum_d[i + 1], M.output_dtype)
            zipX.append((M, fX))
            zipXY.append((M, fX, fY))
        self._zipX = zipX
        self._zipXY = zipXY

    def __call__(self, X) -> "np.ndarray[int, np.dtype[np.number]]":
        Y = np.concatenate([M(f(X)) for M, f in self._zipX], axis=-1)
        return Y

    def p(self, Y: np.ndarray, givenX: np.ndarray) -> np.ndarray:
        probs = np.stack([M.p(fY(Y), givenX=fX(givenX)) for M, fX, fY in self._zipXY])
        return np.prod(probs, axis=0)

    def make_channel(self) -> DiscreteChannel:
        return DiscreteChannel.hstack(*[M.make_channel() for M in self.mechanisms])
