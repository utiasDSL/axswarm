from functools import partial

import jax
import jax.numpy as jp
from jax import Array


@jax.jit
def binomial_coeff(N: int, m: int) -> int:
    K = jp.atleast_1d(jp.minimum(m, N - m))[0]
    return jax.lax.fori_loop(0, K, lambda k, result: result * (N - k) // (k + 1), 1)


@partial(jax.jit, static_argnums=(0, 1))
def bernstein_matrices(K: int, N: int, freq: int) -> tuple[Array, Array, Array]:
    k, n = jp.meshgrid(jp.arange(K), jp.arange(N + 1), indexing="ij")
    W = _bernstein_coeff(k[..., None], n[..., None], K, N, freq)[..., 0]
    dW = _bernstein_dcoeff(k[..., None], n[..., None], K, N, freq)[..., 0]
    ddW = _bernstein_ddcoeff(k[..., None], n[..., None], K, N, freq)[..., 0]
    return _expand_coeff(W), _expand_coeff(dW), _expand_coeff(ddW)


@jax.jit
def bernstein_input(W: Array, dW: Array) -> Array:
    return jp.concat(
        (W.reshape(-1, W.shape[1] * 3), dW.reshape(-1, dW.shape[1] * 3)), axis=-1
    ).reshape(-1, W.shape[1])


@partial(jp.vectorize, signature="(1),(1)->(1)", excluded=[2, 3, 4])
def _bernstein_coeff(k: Array, n: Array, K: int, N: int, freq: int) -> Array:
    assert k.shape == (1,)
    t, T = k / freq, (K - 1) / freq
    return (t**n) * binomial_coeff(N, n) * ((T - t) ** (N - n)) / T**N


@partial(jp.vectorize, signature="(1),(1)->(1)", excluded=[2, 3, 4])
def _bernstein_dcoeff(k: Array, n: Array, K: int, N: int, freq: int) -> Array:
    t, T = k / freq, (K - 1) / freq
    dw = (
        T**-N
        * binomial_coeff(N, n)
        * (n * (t ** (n - 1)) * ((T - t) ** (N - n)) - (t**n) * (N - n) * ((T - t) ** (N - n - 1)))
    )
    dw = jp.where((k == 0) & (n == 0), -N / T, dw)
    dw = jp.where((k == (K - 1)) & (n == N), N / T, dw)
    return dw


@partial(jp.vectorize, signature="(1),(1)->(1)", excluded=[2, 3, 4])
def _bernstein_ddcoeff(k: Array, n: Array, K: int, N: int, freq: int) -> Array:
    t, T = k / freq, (K - 1) / freq
    ddw = (
        (T**-N)
        * binomial_coeff(N, n)
        * (
            n * (n - 1) * (t ** (n - 2)) * ((T - t) ** (N - n))
            - 2 * n * (N - n) * (t ** (n - 1)) * ((T - t) ** (N - n - 1))
            + (N - n) * (N - n - 1) * (t**n) * ((T - t) ** (N - n - 2))
        )
    )
    ddw = jp.where((k == 0) & (n == 0), N * (N - 1) / (T**2), ddw)
    ddw = jp.where((k == (K - 1)) & (n == N), N * (N - 1) / (T**2), ddw)
    ddw = jp.where((k == 0) & (n == 1), -2 * N * (N - 1) / (T**2), ddw)
    ddw = jp.where((k == (K - 1)) & (n == N - 1), -2 * N * (N - 1) / (T**2), ddw)
    return ddw


def _expand_coeff(W: Array) -> Array:
    K, N = W.shape[0], W.shape[1]
    expanded_W = jp.empty((3 * K, 3 * N))
    k_idx = jp.repeat(3 * jp.arange(K)[:, None], N, axis=1)
    n_idx = jp.tile(jp.arange(N), (K, 1))
    expanded_W = expanded_W.at[k_idx, n_idx].set(W)
    expanded_W = expanded_W.at[k_idx + 1, n_idx + N].set(W)
    expanded_W = expanded_W.at[k_idx + 2, n_idx + 2 * N].set(W)
    return expanded_W
