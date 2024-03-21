from typing import Any, Literal, cast, List as List_, Union
import numpy as np
from scipy.spatial import KDTree
import scipy.stats
import scipy.special
import pandas as pd
import abc
import numpy as np
import scipy.stats
import scipy.special
import abc


NumArray = Union[List_, np.ndarray, pd.Series, pd.DataFrame]


# -----------------------------


class ColumnGroupEmbedding(abc.ABC):
    scale: float
    n_columns: int
    d: int  # Number of dimensions of the embedding space

    def __repr__(self):
        return f"{self.__class__.__name__}(n_columns={self.n_columns}, scale={self.scale}, d={self.d})"

    def __init__(self, orig: NumArray, scale=1.0):
        # Shapes accepted: (-1, d) and (-1) for d=1
        orig = np.asarray(orig)
        assert 1 <= len(orig.shape) <= 2, orig.shape
        self.orig2D, self._d1_implicit = (
            (orig, False) if len(orig.shape) == 2 else (orig[:, None], True)
        )
        self.n_columns = self.orig2D.shape[1]
        self.d = self.n_columns
        self.scale = scale

    def embed(self, obs: np.ndarray):
        """
        Shapes accepted if n_columns==1: () and (-1) and (-1, 1)
        Shapes accepted if n_columns>1: (-1,n_columns) and (n_columns)
        """
        # Parse input ensuring standard shape (n, n_columns)
        obs = np.asarray(obs)
        if self.n_columns == 1:
            assert (
                0 <= len(obs.shape) <= 2
            ), f"Expected shapes () or (-1) or (-1, 1). Got: {obs.shape}"
            obs2D = obs
            obs2D = obs[:, None] if len(obs.shape) == 1 else np.atleast_2d(obs)
        else:
            assert (
                1 <= len(obs.shape) <= 2
            ), f"Expected shapes (-1, n_columns) or (n_columns), where n_columns={self.n_columns}. Got: {obs.shape}"
            obs2D = np.atleast_2d(obs)
        # Compute embedding
        return self._embed(obs2D)

    @abc.abstractmethod
    def _embed(self, obs: np.ndarray):
        raise NotImplementedError

    def unembed(self, emb: np.ndarray):
        # Input has standard shape (n, d_embed)
        obs2D = self._unembed(emb)
        # Reshape output according to orig data
        obs = obs2D[:, 0] if self._d1_implicit else obs2D
        return obs

    @abc.abstractmethod
    def _unembed(self, emb: np.ndarray):
        raise NotImplementedError


class ContinuousColumnGroupEmbedding(ColumnGroupEmbedding):

    def __init__(self, orig: NumArray, scale=1.0):
        super().__init__(orig, scale)
        orig2D = self.orig2D
        self.mu = orig2D.mean(axis=0)
        self.sigma = orig2D.std(axis=0)

    def _embed(self, obs: np.ndarray):
        emb = (obs - self.mu) * (self.scale / self.sigma)
        return emb

    def _unembed(self, emb: np.ndarray):
        obs = emb * (self.sigma / self.scale) + self.mu
        return obs


class DiscreteColumnGroupEmbedding(ContinuousColumnGroupEmbedding):

    def __init__(self, orig: NumArray, scale=1.0):
        super().__init__(orig, scale=scale)
        # Scale data as if it was continuous:
        scaled_orig = self.embed(self.orig2D)
        # Prepare neighbors:
        self.uniques = np.unique(scaled_orig, axis=0)
        self.tree = KDTree(self.uniques)

    def _unembed(self, emb: np.ndarray):
        # Find nearest embedding neighbor
        _, idx = self.tree.query(emb, k=1)
        emb = self.uniques[idx]
        # Undo scaling:
        obs = super()._unembed(emb)
        return obs


class CategoricalColumnGroupEmbedding(ColumnGroupEmbedding):

    def __init__(self, orig: NumArray, n_cat: Union[int, None] = None, scale=1.0):
        super().__init__(orig, scale)
        X_cat = self.orig2D
        assert X_cat.shape[-1] == 1, X_cat.shape
        n_cat = int(X_cat.max() + 1) if n_cat is None else n_cat
        assert np.all(0 <= X_cat), X_cat.min()
        assert np.all(X_cat <= n_cat - 1), (X_cat.max(), n_cat)
        assert np.allclose(X_cat, np.round(X_cat)), X_cat[
            np.argmax(np.abs(X_cat - np.round(X_cat)))
        ]
        simplex = CategoricalSimplex(n_cat, lin_scale=1)
        # Awful temporary fix: The simplex should allow me to set the side scale directly
        side_scale = (
            self.scale * simplex.lin_scale["radius"] / simplex.lin_scale["side"]
        )
        self.simplex = CategoricalSimplex(n_cat, lin_scale=side_scale)  # type:ignore
        assert self.simplex.lin_scale["side"] == self.scale, (
            simplex.lin_scale["side"],
            self.scale,
        )
        self.n_cat = n_cat
        self.d = n_cat - 1  # Overwrite self.d

    def _embed(self, orig_cat: np.ndarray):
        emb = self.simplex.cat_to_lin(orig_cat)
        return emb

    def _unembed(self, emb: np.ndarray):
        # Rescaling commented because the argmax is unaffected with it
        obs = self.simplex.lin_to_cat(emb)
        return obs


class MixedColumnGroupEmbedding(ColumnGroupEmbedding):

    _d1_implicit = False
    types: List_[ColumnGroupEmbedding]

    def __init__(self, *groups: ColumnGroupEmbedding):
        self.groups = groups
        self.n_columns = sum(gr.n_columns for gr in groups)
        self.d = sum(gr.d for gr in groups)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_columns={self.n_columns}, d={self.d}, groups={', '.join(repr(gr) for gr in self.groups)})"

    def _embed(self, obs: np.ndarray):
        # Split by columns and embed each group
        i = 0
        embs = []
        for gr in self.groups:
            j = i + gr.n_columns
            emb = gr._embed(obs[:, i:j])
            embs.append(emb)
            i = j
        emb = np.concatenate(embs, axis=1)
        return emb

    def _unembed(self, emb: np.ndarray):
        obs_ = []
        i = 0
        for gr in self.groups:
            j = i + gr.d
            obs = gr._unembed(emb[:, i:j])
            if len(obs.shape) == 1:
                obs = obs[:, None]
            obs_.append(obs)
            i = j
        obs = np.concatenate(obs_, axis=1)
        return obs


class Embedding(ColumnGroupEmbedding):  # Namespace with shorter name for convenience
    Categorical = CategoricalColumnGroupEmbedding
    Continuous = ContinuousColumnGroupEmbedding
    Discrete = DiscreteColumnGroupEmbedding
    Mixed = MixedColumnGroupEmbedding


class CategoricalSimplex:
    """
    Embedding for a categorical variable with d categories. There are three types of embeddings:
    - The "categorical space" (cat), where each category is an integer from 0 to d-1
    - The "one-hot space" (oh) or "affine space", where each category is a one-hot vector (in d-dimensions)
    - The "linear space" (lin), where each category is a point in (d-1)-dimensions
    """

    def __init__(
        self,
        n_cat: int,
        lin_scale: Union[float, Literal["unit_side", "match_oh"]] = "match_oh",
        oh_scale: Union[float, Literal["unit_side", "match_lin"]] = 1,
    ):
        """
        lin_scale: float or 'unit_side' or 'match_oh'
            Radius of the simplex in the linear space.
        oh_scale: float or 'unit_side' or 'match_lin'
            Norm of the one-hot-encoded vectors in the one-hot space.
        """
        self.n_cat = n_cat
        self.d = d = n_cat - 1  # Dimensionality of the simplex

        # Default scales. Caution: side and radius scale simultaneously but not vertex_norm because of the affine map
        dt_scales = np.dtype(
            [
                ("vertex_norm", float),
                ("side", float),
                ("radius", float),
                ("factor", float),
            ]
        )
        Scale = lambda values, by: np.array(
            tuple(by * np.array(values.tolist())), dtype=dt_scales
        )
        s_oh = Scale(np.array([1, np.sqrt(2), np.sqrt(d / (d + 1)), 1]), 1.0)
        s_lin = Scale(np.array([1, s_oh["side"] / s_oh["radius"], 1, 1]), 1.0)
        s_lin["factor"] = 1

        # Parse the scale arguments
        f = lambda x, ref: cast(
            Union[float, None],
            1 / ref["side"] if x == "unit_side" else None if isinstance(x, str) else x,
        )
        _s_lin = f(lin_scale, s_lin)
        _s_oh = f(oh_scale, s_oh)
        if _s_lin is None or _s_oh is None:
            if _s_lin is not None:
                s_lin = Scale(s_lin, _s_lin)
                s_oh = Scale(s_oh, s_lin["radius"] / s_oh["radius"])
            elif _s_oh is not None:
                s_oh = Scale(s_oh, _s_oh)
                s_lin = Scale(s_lin, s_oh["radius"] / s_lin["radius"])
            else:
                raise Exception("Cannot have match_oh and match_lin simultaneously")
        else:
            s_lin = Scale(s_lin, _s_lin)
            s_oh = Scale(s_oh, _s_oh)
        self.lin_scale = s_lin
        self.oh_scale = s_oh

        V = self._simplex_vertices(d)  # of radius 1.
        self.isometry = np.sqrt(d / (d + 1)) * V

        self.lin_vertices = self.lin_scale["vertex_norm"] * V
        self.oh_vertices = self.oh_scale["vertex_norm"] * np.eye(d + 1)
        self.oh_centroid = np.mean(self.oh_vertices, axis=0)
        self.lin_volume = (
            np.sqrt((d + 1) / 2**d)
            / scipy.special.gamma(1 + d)
            * self.lin_scale["side"] ** d
        )
        self.oh_volume = (
            np.sqrt((d + 1) / 2**d)
            / scipy.special.gamma(1 + d)
            * self.oh_scale["side"] ** d
        )

    @staticmethod
    def _simplex_vertices(d: int):
        # The columns of V are the vertices of the d-dimensional simplex of radius 1
        assert d > 0
        V = np.array([[-1, 1]])
        k = 2
        while k <= d:
            V_i = [([-1 / k, *(np.sqrt(k**2 - 1) / k * u)]) for u in V.T]
            v_k = [1, *np.zeros(k - 1)]
            V = np.array([v_k, *V_i]).T
            k += 1
        return V

    def _clean_cat(self, X_cat: np.ndarray):
        assert np.all(np.isclose(X_cat, X_cat.astype(int))), X_cat
        X_cat = X_cat.astype(int)
        if len(X_cat.shape) == 2 and X_cat.shape[-1] == 1:
            X_cat = X_cat[:, 0]
        return X_cat

    def cat_to_oh(self, X_cat: np.ndarray):
        return self.oh_vertices.T[self._clean_cat(X_cat)]

    def cat_to_lin(self, X_cat: np.ndarray):
        return self.lin_vertices.T[self._clean_cat(X_cat)]

    def oh_to_cat(self, X: np.ndarray):
        return np.argmax(X, axis=-1)

    def lin_to_cat(self, X: np.ndarray):
        X_OH = self.lin_to_oh(X)
        return self.oh_to_cat(X_OH)

    def oh_to_lin(self, X_oh: np.ndarray):
        # X_oh has rows of one-hot vectors
        X_oh = X_oh / self.oh_scale["side"]
        X_lin = np.matmul(X_oh, self.isometry.T)
        return X_lin * self.lin_scale["side"]

    def lin_to_oh(self, X_lin: np.ndarray):
        X_lin = X_lin / self.lin_scale["side"]
        X_oh = np.matmul(X_lin[..., None, :], self.isometry)
        return X_oh * self.oh_scale["side"] + self.oh_centroid
