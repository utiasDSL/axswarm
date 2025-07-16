from __future__ import annotations

import jax
import jax.numpy as jp
from flax.struct import dataclass
from jax import Array


@dataclass
class EqualityConstraint:
    G: Array
    h: Array
    _G_T_G: Array
    _G_T_h: Array
    mask: Array
    tol: Array

    @staticmethod
    def init(G: Array, h: Array, tol: float = 1e-2) -> EqualityConstraint:
        G_T_G = G.mT @ G
        G_T_h = G.mT @ h
        mask = jp.ones(shape=G.shape[:-2], dtype=bool)
        assert G.shape[-2] == h.shape[-1]
        assert G_T_G.shape[-1] == G_T_G.shape[-2] == G.shape[-1]
        assert G_T_h.shape[-1] == G.shape[-1]
        return EqualityConstraint(G, h, G_T_G, G_T_h, mask, jp.array(tol))

    @staticmethod
    def quadratic_term(cnstr: EqualityConstraint) -> Array:
        return cnstr._G_T_G * cnstr.mask

    @staticmethod
    def linear_term(cnstr: EqualityConstraint) -> Array:
        return -cnstr._G_T_h * cnstr.mask

    @staticmethod
    @jax.jit
    def satisfied(cnstr: EqualityConstraint, x: Array) -> Array:
        return (jp.max(jp.abs(cnstr.G @ x - cnstr.h), axis=-1) <= cnstr.tol) | ~cnstr.mask


@dataclass
class InequalityConstraint:
    G: Array
    h: Array
    _G_T_G: Array
    _G_T_h: Array
    slack: Array
    mask: Array
    active: Array
    tol: float
    active_range: float

    @staticmethod
    def init(G: Array, h: Array, tol: float, active_range: float) -> InequalityConstraint:
        G_T_G = G.T @ G
        G_T_h = G.T @ h
        slack = jp.zeros_like(h)
        mask = jp.ones(shape=G.shape[:-2], dtype=bool)
        assert G.shape[-2] == h.shape[-1]
        assert G_T_G.shape[-1] == G_T_G.shape[-2] == G.shape[-1]
        assert G_T_h.shape[-1] == G.shape[-1]
        active = jp.ones_like(mask, dtype=bool)
        return InequalityConstraint(G, h, G_T_G, G_T_h, slack, mask, active, tol, active_range)

    @staticmethod
    def quadratic_term(cnstr: InequalityConstraint) -> Array:
        return cnstr._G_T_G * cnstr.mask * cnstr.active

    @staticmethod
    def linear_term(cnstr: InequalityConstraint) -> Array:
        return (-cnstr._G_T_h + cnstr.G.T @ cnstr.slack) * cnstr.mask * cnstr.active

    @staticmethod
    def update(cnstr: InequalityConstraint, x: Array) -> InequalityConstraint:
        slack = jp.maximum(0, -cnstr.G @ x + cnstr.h)
        active = jp.any(cnstr.G @ x - cnstr.h > cnstr.active_range, axis=-1)
        return cnstr.replace(slack=slack, active=active)

    @staticmethod
    def satisfied(cnstr: InequalityConstraint, x: Array) -> bool:
        return (jp.max(cnstr.G @ x - cnstr.h, axis=-1) <= cnstr.tol) | ~cnstr.mask


@dataclass
class PolarInequalityConstraint:
    """Polar inequality constraint of a specialized form.

    The constraint is defined in polar coordinates as
    Gx + c = h(alpha, beta, d)

    with the boundary condition lwr_bound <= d <= upr_bound. Here, 'alpha', 'beta', and 'd' are
    vectors with a length of K+1, where 'd' represents the distance from the origin, 'alpha' the
    azimuthal angle, and 'beta' the polar angle. The vector 'h' has a length of 3 * (K+1), where
    each set of three elements in 'h()' represents a point in 3D space expressed as:

    d[k] * [cos(alpha[k]) * sin(beta[k]), sin(alpha[k]) * sin(beta[k]), cos(beta[k])]^T

    This represents a unit vector defined by angles 'alpha[k]' and 'beta[k]', scaled by 'd[k]',
    where 'k' is an index running from 0 to K. The index range from 0 to K can be interpreted as
    discrete time steps, allowing this constraint to serve as a Barrier Function (BF) constraint.

    Note:
        bf_gamma is fixed to 1.0 in our implementation which enables a faster computation of the
        update.
    """

    G: Array
    c: Array
    h: Array
    _G_T_G: Array
    lwr_bound: float | None
    upr_bound: float | None
    mask: Array
    active: Array
    tol: float
    active_range: float

    @staticmethod
    def init(
        G: Array,
        c: Array,
        lwr_bound: float | None = None,
        upr_bound: float | None = None,
        tol: float = 1e-2,
        mask: Array | None = None,
        active_range: float = 1.0,
    ) -> PolarInequalityConstraint:
        if mask is None:
            mask = jp.ones(shape=G.shape[:-2], dtype=bool)
        else:
            assert mask.shape == G.shape[:-2]
        active = jp.ones_like(mask, dtype=bool)
        return PolarInequalityConstraint(
            G, c, -c, G.mT @ G, lwr_bound, upr_bound, mask, active, tol, active_range
        )

    @staticmethod
    @jax.jit
    def quadratic_term(cnstr: PolarInequalityConstraint) -> Array:
        mask, active = cnstr.mask[..., None, None], cnstr.active[..., None, None]
        return cnstr._G_T_G * mask * active

    @staticmethod
    @jax.jit
    def linear_term(cnstr: PolarInequalityConstraint) -> Array:
        mask, active = cnstr.mask[..., None], cnstr.active[..., None]
        return (-cnstr.G.mT @ cnstr.h[..., None])[..., 0] * mask * active

    @staticmethod
    @jax.jit
    def update(cnstr: PolarInequalityConstraint, x: Array) -> PolarInequalityConstraint:
        if cnstr.G.shape[-1] != x.shape[-1]:
            raise ValueError("G and x are not compatible sizes")
        assert not (cnstr.upr_bound is not None and cnstr.lwr_bound is not None)

        h = cnstr.G @ x + cnstr.c
        h = h.reshape(*h.shape[:-1], -1, 3)
        h_norm = jp.linalg.norm(h, axis=-1, keepdims=True)

        if cnstr.upr_bound is not None:
            mask = h_norm > cnstr.upr_bound
            bound = cnstr.upr_bound
        elif cnstr.lwr_bound is not None:
            mask = h_norm < cnstr.lwr_bound
            bound = cnstr.lwr_bound
        else:
            raise ValueError("Must be either upper or lower")
        h = jp.where(mask, h / h_norm * bound, h)
        h = h.reshape(*h.shape[:-2], -1) - cnstr.c
        active = jp.any(cnstr.G @ x - h > cnstr.active_range, axis=-1)
        return cnstr.replace(h=h, active=active)

    @staticmethod
    @jax.jit
    def satisfied(cnstr: PolarInequalityConstraint, x: Array) -> Array:
        return (jp.max(jp.abs(cnstr.G @ x - cnstr.h), axis=-1) <= cnstr.tol) | ~cnstr.mask
