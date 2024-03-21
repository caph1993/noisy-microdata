import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
from typing import Any, Literal, cast, List as List_, Union
import warnings
import abc
import numpy as np
import scipy.stats
import scipy.special
import abc


cast_rv_continuous = lambda x: cast(scipy.stats.rv_continuous, x)

IntOrArray = Union[int, np.ndarray]
FloatOrArray = Union[float, np.ndarray]


class RadialSpace:
    "Hyperspheres in R^d"

    @classmethod
    def d_volume(cls, d: IntOrArray, r: FloatOrArray = 1):
        "Volume enclosed by the hypersphere in d dimensions with radius r"
        return np.pi ** (d / 2) / scipy.special.gamma(1 + d / 2) * r**d

    @classmethod
    def d_surface(cls, d: IntOrArray, r: FloatOrArray = 1):
        "Surface of the hypersphere in d dimensions with radius r, i.e. derivative of volume(r) wrt r"
        return np.pi ** (d / 2) / scipy.special.gamma(1 + d / 2) * r ** (d - 1) * d

    _cache_instances = {}  # class attribute to cache instances

    def __new__(cls, d: int):
        key = (cls.__name__, d)
        obj = cls._cache_instances.get(key)
        if obj is not None:
            return obj
        obj = super().__new__(cls)
        obj._init(d)
        cls._cache_instances[key] = obj
        return obj

    def __repr__(self):
        return f"{self.__class__.__name__}({self.d})"

    def _init(
        self, d: int
    ):  # if it was called __init__, it would be called even if cached
        assert d == round(d) and d >= 1, d
        self.d = d
        "Mass of a unit hypersphere in d dimensions with constant unit density"
        self._unit_vol = self.d_volume(1)
        self._unit_surface = self.d_surface(1)

    def volume(self, r: FloatOrArray = 1):
        return self.d_volume(self.d, r)

    def surface(self, r: FloatOrArray = 1):
        return self.d_surface(self.d, r)

    def random_on_surface(self, r: FloatOrArray = 1, size=None):
        "Draw uniformly from the sphere surface"
        r = np.asarray(r).astype(float)
        shape = r.shape if size is None else tuple(np.ravel(size))
        X = np.random.normal(0, 1, size=(*shape, self.d))
        X /= r * np.linalg.norm(X, axis=-1, keepdims=True)
        return X

    def surface_marginal_pdf(self, r: FloatOrArray = 1, d: Union[int, None] = None):
        "Draw uniformly from the sphere surface and take the first coordinate, ignoring the rest"
        r = np.asarray(r).astype(float)
        d = self.d if d is None else d
        rv = cast_rv_continuous(scipy.stats.beta((d - 1) / 2, (d - 1) / 2))
        return rv.pdf((r + 1) / 2) / 2

    def alt_random_surface_marginal(self, r: FloatOrArray = 1, size=None, d=None):
        "Alternative to self.random(r, size)[:,0]"
        r = np.asarray(r).astype(float)
        d = self.d if d is None else d
        if d == 1:  # (special case: Rademacher distribution)
            return (2 * np.random.randint(0, 2, size=size) - 1.0) * r
        rv = cast_rv_continuous(scipy.stats.beta((d - 1) / 2, (d - 1) / 2))
        return (rv.rvs(size=size) * 2 - 1) * r


# -----------------------------


class IstotropicDistribution:
    """
    The total mass is 1 = integral of norm_pdf(r) dr.
    """

    _r_plot: float  # large radius for plotting purposes

    def __init__(self, d: int, r: float = 1.0):
        """
        d: int (dimensionality)
        r: float (radial scale, default 1.0)
        """
        assert d == round(d) and d >= 1, d
        self.space = RadialSpace(d)
        self.d = d
        self.r = r
        self._r_plot = self.r * self.d**0.5 * 2
        self.fuzzy_radius = np.nan  # Must be set by each subclass
        self._post_init()

    @classmethod
    def from_fuzzy_radius(cls, d: int, fuzzy_radius: float):
        "Create a distribution with a given fuzzy_radius"
        r = fuzzy_radius / cls(d, 1).fuzzy_radius
        # # Reminder: Alternative using scipy solver:
        # f = lambda r: cls(d, r).fuzzy_radius - fuzzy_radius
        # r = scipy.optimize.root_scalar(f, bracket=[1e-10, 1e10]).root
        obj = cls(d, r)
        assert np.isclose(obj.fuzzy_radius, fuzzy_radius), (
            obj.fuzzy_radius,
            fuzzy_radius,
        )
        return obj

    def _post_init(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.d}, r={self.r})"

    @abc.abstractmethod
    def norm_pdf(self, r: FloatOrArray):
        "Density of |X| for X~self. (r>=0)"
        raise NotImplementedError

    def norm_cdf(self, r: FloatOrArray):
        "probability mass that is enclosed in the hypersphere of radius r. (r>=0)"
        raise NotImplementedError

    def random(self, size=None):
        "Draw a sample from the distribution"
        X = self.space.random_on_surface(1, size=size)
        X *= self.random_norm(size)[..., None]
        return X

    @abc.abstractmethod
    def random_norm(self, size=None):
        raise NotImplementedError

    def pointwise_irradiance(self, r: FloatOrArray):
        """
        Equals norm_pdf(r) / space.surface(r)
        This is the "euclidean density" in the following sense:
        Fix any point x at radius r, let B(l) be the hypercube of side l centered at x.
        The pointwise_irradiance is the limit as l goes to 0 of the integral of the distribution
        over B(l) divided by the norm_cdf of l.
        """
        r = np.where(r == 0, 1e-20, r)  # Correction for singularity of some rvs
        return self.norm_pdf(r) / self.space.surface(r)

    def fuzzy_membership(self, r: FloatOrArray):
        # Returns a number in [0,1]
        return self.pointwise_irradiance(r) / self.pointwise_irradiance(0)

    def _ref_marginal(self, r: FloatOrArray):
        "Just for plotting purposes"
        raise NotImplementedError

    def _ref_irradiance(self, r: FloatOrArray) -> FloatOrArray:
        "Just for plotting purposes"
        raise NotImplementedError

    def marginal_pdf(self, x: FloatOrArray) -> FloatOrArray:
        "Just for plotting purposes"
        raise NotImplementedError

    def _compute_marginal_pdf(self, x: float) -> float:
        "Explicit computation in case no formula is available"
        warnings.warn(f"May have bugs. Not tested yet!")
        f = lambda r: self.pointwise_irradiance(
            (r**2 + x**2) ** 0.5
        ) * self.space.d_surface(self.d - 1, r)
        return scipy.integrate.quad(f, 0, np.inf)[0]

    def random_marginal(self, r: FloatOrArray = 1, size=None):
        "Alternative to self.random(r, size)[:,0]"
        r = np.asarray(r).astype(float)
        return self.space.alt_random_surface_marginal(size=size) * self.random_norm(
            size=size
        )


class ScipyIsotropicDistribution(IstotropicDistribution):
    _rv: scipy.stats.rv_continuous

    def random_norm(self, size=None):
        return self._rv.rvs(size) * self.r

    def norm_pdf(self, r: FloatOrArray):
        "Density for the prob. that a point chosen uniformly from the hypersphere (norm_cdf) has radius r. (r>=0)"
        return self._rv.pdf(r / self.r) / self.r

    def norm_cdf(self, r: FloatOrArray):
        return self._rv.cdf(r / self.r)

    def norm_rms(self):
        return (self.r**2 * self._rv.moment(2)) ** 0.5

    def norm_mean(self):
        return self._rv.moment(1) * self.r


class UniformIsotropicDistribution(ScipyIsotropicDistribution):
    def _post_init(self):
        super()._post_init()
        self._rv = cast_rv_continuous(scipy.stats.beta(self.d, 1))
        self._rv_marginal = cast_rv_continuous(
            scipy.stats.beta((self.d + 1) / 2, (self.d + 1) / 2)
        )
        self._r_plot = self.r * 1.2
        self.fuzzy_radius = self.r

    def _ref_irradiance(self, r: FloatOrArray):
        return np.where(r < self.r, 1, 0) / self.space.volume(self.r)

    def marginal_pdf(self, x: FloatOrArray):
        # Reminder. This is equivalent:
        # return self.space.surface_marginal_pdf(x, d=self.d+2) / self.r
        x = np.clip(x / self.r, -1, 1)
        return self._rv_marginal.pdf((x + 1) / 2) / self.r / 2

    def random_marginal(self, size=None):
        # Alternative implementation
        return (self._rv_marginal.rvs(size=size) * 2 - 1) * self.r

    # Reminder: Random_radius alternatives:
    # Formula 1: np.random.random(size)**(1/self.d) * self.r
    # Formula 2: np.max(np.random.random((*shape, d)), axis=1) * self.r


class ExponentialIsotropicDistribution(ScipyIsotropicDistribution):

    def _post_init(self):
        super()._post_init()
        self._rv = cast_rv_continuous(scipy.stats.gamma(self.d))
        self._r_plot = self.r * self.d**0.5 * 8
        irr_rv = cast_rv_continuous(scipy.stats.expon())
        self.fuzzy_radius = self.r / irr_rv.pdf(0)

    def _ref_irradiance(self, r: FloatOrArray):
        c = scipy.special.gamma(self.d) * self.space.surface(self.r)
        return 1 / self.r * np.exp(-r / self.r) / c

    def marginal_pdf(self, x: FloatOrArray):
        # if self.d == 1: # Reminder: this code would be equivalent for d=1
        #     return np.exp(-np.abs(x)/self.r) / (2*self.r)
        d = self.d
        G = scipy.special.gamma
        c = 2 ** (d / 2) / np.pi * G(1 + d / 2) / G(1 + d)
        f = lambda x: scipy.special.kv(d / 2, x) * x ** (d / 2)
        return c * f(np.abs(x) / self.r) / self.r


class GaussianIsotropicDistribution(ScipyIsotropicDistribution):

    def _post_init(self):
        super()._post_init()
        self._rv = cast_rv_continuous(scipy.stats.chi(self.d))
        self._r_plot = self.r * self.d**0.5 * 5
        irr_rv = cast_rv_continuous(scipy.stats.halfnorm())
        self.fuzzy_radius = self.r / irr_rv.pdf(0)

    def random(self, size=None):
        shape = () if size is None else tuple(np.ravel(size))
        X = np.random.normal(0, 1, size=(*shape, self.d))
        return X * self.r

    def pointwise_irradiance(self, r: FloatOrArray):
        irr_rv = cast_rv_continuous(scipy.stats.halfnorm())
        c = 2 ** (self.d / 2 + 1) / 5 * scipy.special.gamma(self.d / 2)
        c *= self.space.surface(self.r)
        return irr_rv.pdf(r / self.r) / self.r / c

    def fuzzy_membership(self, r: FloatOrArray):
        irr_rv = cast_rv_continuous(scipy.stats.halfnorm())
        return irr_rv.pdf(r / self.r) / irr_rv.pdf(0)

    def marginal_pdf(self, x: FloatOrArray):
        return scipy.stats.norm.pdf(x / self.r) / self.r
