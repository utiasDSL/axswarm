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
    active: Array
    tol: Array

    @staticmethod
    def init(G: Array, h: Array, tol: float = 1e-2) -> EqualityConstraint:
        G_T_G = G.mT @ G
        G_T_h = G.mT @ h
        active = jp.ones(shape=G.shape[:-2], dtype=bool)
        assert G.shape[-2] == h.shape[-1]
        assert G_T_G.shape[-1] == G_T_G.shape[-2] == G.shape[-1]
        assert G_T_h.shape[-1] == G.shape[-1]
        return EqualityConstraint(G, h, G_T_G, G_T_h, active, jp.array(tol))

    @staticmethod
    def quadratic_term(cnstr: EqualityConstraint) -> Array:
        return cnstr._G_T_G * cnstr.active

    @staticmethod
    def linear_term(cnstr: EqualityConstraint) -> Array:
        return -cnstr._G_T_h * cnstr.active

    @staticmethod
    @jax.jit
    def satisfied(cnstr: EqualityConstraint, zeta: Array) -> Array:
        return (jp.max(jp.abs(cnstr.G @ zeta - cnstr.h), axis=-1) <= cnstr.tol) | ~cnstr.active


@dataclass
class InequalityConstraint:
    G: Array
    h: Array
    _G_T_G: Array
    _G_T_h: Array
    slack: Array
    active: Array
    tol: float = 1e-2

    @staticmethod
    def init(G: Array, h: Array, tol: float = 1e-2) -> InequalityConstraint:
        G_T_G = G.T @ G
        G_T_h = G.T @ h
        slack = jp.zeros_like(h)
        active = jp.ones(shape=G.shape[:-2], dtype=bool)
        assert G.shape[-2] == h.shape[-1]
        assert G_T_G.shape[-1] == G_T_G.shape[-2] == G.shape[-1]
        assert G_T_h.shape[-1] == G.shape[-1]
        return InequalityConstraint(G, h, G_T_G, G_T_h, slack, active, tol)

    @staticmethod
    def quadratic_term(cnstr: InequalityConstraint) -> Array:
        return cnstr._G_T_G * cnstr.active

    @staticmethod
    def linear_term(cnstr: InequalityConstraint) -> Array:
        return (-cnstr._G_T_h + cnstr.G.T @ cnstr.slack) * cnstr.active

    @staticmethod
    def update(cnstr: InequalityConstraint, x: Array) -> InequalityConstraint:
        slack = jp.maximum(0, -cnstr.G @ x + cnstr.h)
        return cnstr.replace(slack=jp.where(cnstr.active, slack, cnstr.slack))

    @staticmethod
    def satisfied(cnstr: InequalityConstraint, zeta: Array) -> bool:
        return (jp.max(cnstr.G @ zeta - cnstr.h, axis=-1) <= cnstr.tol) | ~cnstr.active


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
    active: Array
    tol: float

    @staticmethod
    def init(
        G: Array,
        c: Array,
        lwr_bound: float | None = None,
        upr_bound: float | None = None,
        tol: float = 1e-2,
        active: Array | None = None,
    ) -> PolarInequalityConstraint:
        if active is None:
            active = jp.ones(shape=G.shape[:-2], dtype=bool)
        else:
            assert active.shape == G.shape[:-2]
        return PolarInequalityConstraint(G, c, -c, G.mT @ G, lwr_bound, upr_bound, active, tol)

    @staticmethod
    @jax.jit
    def quadratic_term(cnstr: PolarInequalityConstraint) -> Array:
        return cnstr._G_T_G * cnstr.active[..., None, None]

    @staticmethod
    @jax.jit
    def linear_term(cnstr: PolarInequalityConstraint) -> Array:
        return (-cnstr.G.mT @ cnstr.h[..., None])[..., 0] * cnstr.active[..., None]

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
        cnstr = cnstr.replace(
            h=jp.where(cnstr.active[..., None], h.reshape(*h.shape[:-2], -1) - cnstr.c, cnstr.h)
        )
        return cnstr

    @staticmethod
    @jax.jit
    def satisfied(cnstr: PolarInequalityConstraint, zeta: Array) -> Array:
        return (jp.max(jp.abs(cnstr.G @ zeta - cnstr.h), axis=-1) <= cnstr.tol) | ~cnstr.active
