from pathlib import Path
import pickle
from typing import Any, Literal, Tuple, cast, List as List_, Union
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import KDTree
from .isotropic import (
    ExponentialIsotropicDistribution,
    GaussianIsotropicDistribution,
    UniformIsotropicDistribution,
)


IntOrArray = Union[int, np.ndarray]
FloatOrArray = Union[float, np.ndarray]
IntOrList = Union[List_[int], int]
NumArray = Union[List_, np.ndarray, pd.Series, pd.DataFrame]
PForPNorm = Union[Literal[1, 2], float]


class Ellipsoid:
    "Ellipse in n-dimensions, defined by its center (mu), amplitudes (lambdas) and orthogonal directions (matrix V)"

    def __init__(
        self,
        mu: np.ndarray,
        lambdas: Union[np.ndarray, float] = 1,
        V: Union[None, np.ndarray] = None,
    ):
        self.mu = mu
        self.d = len(mu)
        if isinstance(lambdas, (int, float)):
            lambdas = np.ones(len(mu)) * lambdas / np.sqrt(len(mu))
        lambdas = np.asarray(lambdas)
        self.lambdas = lambdas
        self.V = V if V is not None else np.eye(len(mu))
        assert mu.shape == lambdas.shape, (mu.shape, lambdas.shape)
        assert self.V.shape == (len(mu), len(mu)), self.V.shape
        self.orientation: Union[None, np.ndarray] = None

    def orient(self, neigh: "Neighborhood"):
        """
        Orient the eigenvectors towards the majority of the data points
        and assign a probability (at least 0.5) of not-inverting for random sampling
        """
        # Project the points onto the ellipsoid to see the direction
        orientation = np.zeros(self.d)
        X = neigh.points - neigh.center
        for i in range(self.d):
            # Count the number of neighbors in the direction of the eigenvector
            dirs = X @ self.V[:, i]
            w_pos = np.sum(neigh.weights[dirs > 0] * dirs[dirs > 0])
            w_neg = np.sum(neigh.weights[dirs < 0] * -dirs[dirs < 0])  # type:ignore
            if np.isclose(w_pos + w_neg, 0):
                orientation[i] = 0.5
                continue
            orientation[i] = w_pos / (w_pos + w_neg + 1e-15)
            if orientation[i] < 0.5:
                self.V[:, i] *= -1
                orientation[i] = 1 - orientation[i]
        self.orientation = orientation

    @classmethod
    def from_neighborhood(cls, neigh: "Neighborhood", forced_center=None, orient=False):
        points, weights = neigh.points, neigh.weights
        if forced_center is not None:
            # Add symmetric points to force the center
            s_points = forced_center - (points - neigh.center)
            points = np.concatenate([points, s_points], axis=0)
            weights = np.concatenate([weights, weights], axis=0)
        mu = np.average(points, axis=0, weights=weights)
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            cov = np.cov(points.T, aweights=weights)
            try:
                diag, V = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                return cls(mu, 0, np.eye(len(mu)))
            diag[np.isclose(diag, 0, atol=1e-15)] = 0
            lambdas = np.sqrt(diag)
            if np.all(lambdas == 0) or np.any(np.isnan(lambdas)):
                return cls(mu, 0, np.eye(len(mu)))
        ell = cls(mu, lambdas, V)
        if orient:
            ell.orient(neigh)
        return ell

    def random(self, p: PForPNorm = 2, size=None, min_scale=None, max_scale=None):
        _N = 1 if size is None else int(np.prod(np.ravel(size)))
        if p == 1:
            dist = ExponentialIsotropicDistribution.from_fuzzy_radius(self.d, 1)
        elif p == 2:
            dist = GaussianIsotropicDistribution.from_fuzzy_radius(self.d, 1)
        else:
            assert p == np.inf, p
            dist = UniformIsotropicDistribution.from_fuzzy_radius(self.d, 1)
        X = dist.random(_N)
        X *= self.lambdas
        if self.orientation is not None:
            sign = 2 * (np.random.random(size=X.shape) < self.orientation[None, :]) - 1
            X = np.abs(X) * sign
        factor = 1
        if min_scale is not None or max_scale is not None:
            assert min_scale is None or max_scale is None or min_scale <= max_scale
            self_scale = np.linalg.norm(self.lambdas)
            if min_scale is not None:
                factor = max(factor, min_scale / self_scale)
            if max_scale is not None and self_scale > 0:
                factor = min(factor, max_scale / self_scale)
        for i in range(_N):
            X[i, :] = (self.V @ X[i, :]) * factor + self.mu
        _size = () if size is None else tuple(np.ravel(size))
        return X.reshape((*_size, self.d))


class NeighborhoodStats:
    "Summary of a Neighborhood instance"

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        p_membership: PForPNorm,
        not_center_count: int,
        not_center_weight: float,
        ell: Union[
            None, Ellipsoid, Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]
        ],
    ):
        self.center = center
        self.radius = radius
        self.not_center_count = not_center_count
        self.not_center_weight = not_center_weight
        self.p_membership = p_membership
        if ell is not None:
            if not isinstance(ell, Ellipsoid):
                lambdas, V, orientation = ell
                ell = Ellipsoid(center, lambdas, V)
                ell.orientation = orientation
            self.ell = ell

    def to_args(self):
        ell_args = (self.ell.lambdas, self.ell.V, self.ell.orientation)
        return (
            self.center,
            self.radius,
            self.p_membership,
            self.not_center_count,
            self.not_center_weight,
            ell_args,
        )


class Neighborhood(NeighborhoodStats):
    "Collection of possibly weighted points within a radius around a central point in R^d"

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        p_membership: PForPNorm,
        points: np.ndarray,
        weights: Union[np.ndarray, None] = None,
        counts: Union[np.ndarray, None] = None,
        centered=False,
    ):
        self.points = points
        self.weights = weights if weights is not None else np.ones(len(points))
        self.counts = counts if counts is not None else np.ones(len(points), dtype=int)
        self.not_center_mask = np.any(self.points != center, axis=-1)
        not_center_count = np.sum(self.counts[self.not_center_mask])
        not_center_weight = cast(float, np.sum(self.weights[self.not_center_mask]))
        super().__init__(
            center, radius, p_membership, not_center_count, not_center_weight, ell=None
        )
        self.ell = Ellipsoid.from_neighborhood(
            self, forced_center=centered, orient=centered
        )


# -----------------------------


import hashlib


class CachedNeighborhoods:

    def __init__(self, emb: np.ndarray, centered=False):
        md5 = hashlib.md5(emb.tobytes()).hexdigest()
        md5 = hashlib.md5(f"{md5}{centered}".encode("ascii")).hexdigest()
        self.tree = KDTree(emb)
        self.filename = f"neighbors_cache_{md5[:10]}.npy"
        self.tmp_filename = f"neighbors_cache_tmp_{md5[:10]}.npy"
        self.centered = centered
        try:
            self.ell_cache = pickle.load(open(self.filename, "rb"))
        except FileNotFoundError:
            self.ell_cache = {}

    def neigh_stats(
        self,
        x: np.ndarray,
        radius: float,
        p_membership: PForPNorm = 2,
        p_query: Union[PForPNorm, Literal["membership"]] = "membership",
        min_neigh=None,
    ) -> NeighborhoodStats:
        if p_query == "membership":
            p_query = p_membership
        args = (radius, p_membership, p_query)
        if min_neigh is None:
            key = str(x)
            bucket = self.ell_cache[key] = self.ell_cache.get(key, {})
            if args not in bucket:
                neigh = self.neigh(x, radius, p_membership, p_query)
                neigh_stats = NeighborhoodStats(*cast(Any, neigh.to_args()))
                bucket[args] = neigh_stats.to_args()
                # Save every 50 new ellipsoids approximately
                if np.random.random() < 0.02:
                    pickle.dump(self.ell_cache, open(self.tmp_filename, "wb"))
                    Path(self.tmp_filename).rename(self.filename)
            else:
                neigh_stats = NeighborhoodStats(*bucket[args])
            return neigh_stats

        # For min_neigh, we need to ensure that the neighborhood has enough neighbors
        def f(r):
            stats = self.neigh_stats(x, r, p_membership, p_query, min_neigh=None)
            return None if stats.not_center_count < min_neigh else stats

        r = radius
        # Try directly with the min_radius
        stats = f(r)
        if stats is not None:
            return stats
        # Find an upper bound quickly for which it works
        r_high = r
        while f(r_high) is None:
            r_high *= 5
        # Use binary search for a tighter upper bound
        r_low = r
        while r_high - r_low > 1e-3:
            r = (r_high + r_low) / 2
            ell = f(r)
            if ell is None:
                r_low = r
            else:
                r_high = r
        stats = f(r_high)
        assert stats is not None
        return stats

    def neigh(
        self,
        x: np.ndarray,
        radius: float,
        p_membership: PForPNorm = 2,
        p_query: Union[PForPNorm, Literal["membership"]] = "membership",
    ):
        if p_query == "membership":
            p_query = p_membership
        d = len(x)
        if p_membership == 2:
            weights_dist = GaussianIsotropicDistribution.from_fuzzy_radius(d, radius)
        elif p_membership == 1:
            weights_dist = ExponentialIsotropicDistribution.from_fuzzy_radius(d, radius)
        elif p_membership == np.inf:
            weights_dist = UniformIsotropicDistribution.from_fuzzy_radius(d, radius)
        # Use r_plot while I implement the ppf properly for all isotropic distributions
        large_radius = weights_dist._r_plot
        idx = self.tree.query_ball_point(x, r=large_radius, p=p_query)
        X = self.tree.data
        X, counts = np.unique(X[idx], axis=0, return_counts=True)
        weights = counts * weights_dist.fuzzy_membership(np.linalg.norm(X - x, axis=1))
        return Neighborhood(x, radius, p_membership, X, weights, counts, self.centered)
