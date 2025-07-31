from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jp
from jax import Array
from numpy.typing import NDArray

from axswarm.constraint import EqualityConstraint, InequalityConstraint, PolarInequalityConstraint
from axswarm.data import SolverData
from axswarm.settings import SolverSettings

ACTIVE_RANGE = -10.0


def solve(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[Array[bool], Array[int], SolverData]:
    # The horizon is dynamically shaped based on which waypoints are in the current horizon. This is
    # therefore the only function we cannot compile with jax.jit.
    with jax.default_device(data.pos.device):
        data = _set_horizon(data, settings)
        # After setting the horizon, everything else is static and hence gets compiled
        return _solve(states, float(t), data, settings)


def _set_horizon(data: SolverData, settings: SolverSettings) -> SolverData:
    start, end, t_discrete = _filter_horizon(
        data.waypoints["time"][data.rank], data.current_time, settings.K, settings.freq
    )
    in_horizon = jp.arange(start, end)
    if len(in_horizon) < 1:
        raise RuntimeError(
            "Error: no waypoints within current horizon. Increase horizon or add waypoints."
        )
    return data.replace(in_horizon=in_horizon, t_discrete=t_discrete)


@jax.jit
def _solve(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[Array[bool], Array[int], SolverData]:
    distances = _compute_swarm_distances(data, settings)
    # Set the initial state and current time
    data = data.replace(x_0=states, current_time=t, distance_matrix=distances)

    # Solve for each drone
    def solve(data: SolverData, rank: Array) -> tuple[SolverData, tuple[bool, int]]:
        data = data.replace(rank=rank)
        success, n_iters, data = _solve_drone(data, settings)
        return data, (success, n_iters)

    data, (success, iters) = jax.lax.scan(solve, data, jp.arange(data.n_drones), data.n_drones)
    return success, iters, data


def _compute_swarm_distances(data: SolverData, settings: SolverSettings) -> Array:
    col = 1.0 / settings.collision_envelope
    distances = jp.linalg.norm((data.pos[None, ...] - data.pos[:, None, ...]) * col, axis=-1)
    distances = jp.where(jp.eye(data.n_drones, dtype=bool)[..., None], jp.inf, distances)
    return distances


def _solve_drone(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Main solve function to be called by user."""
    data = _reset_cost_matrices(data)
    data = _reset_constraints(data)
    data = _add_constraints(data, settings)
    success, iters, data = _am_solve(data, settings)
    data = _spline2states(data, settings)
    # Ensure constraints are None so pytree stays consistent for jax.lax.scan
    data = _reset_constraints(data)
    return success, iters, data


def _reset_cost_matrices(data: SolverData) -> SolverData:
    Q_init = data.quad_cost.at[...].set(data.quad_cost_init)
    q_init = jp.zeros_like(data.linear_cost)
    return data.replace(quad_cost=Q_init, linear_cost=q_init)


def _reset_constraints(data: SolverData) -> SolverData:
    """Reset constraints to initial values"""
    return data.replace(
        pos_constraint=None,
        vel_constraint=None,
        acc_constraint=None,
        input_continuity_constraint=None,
        max_pos_constraint=None,
        max_vel_constraint=None,
        max_acc_constraint=None,
        collision_constraints=None,
    )


def _add_constraints(data: SolverData, settings: SolverSettings) -> SolverData:
    """Setup optimization problem before solving.

    Override of AMSolver method that configures constraints and cost functions.
    """
    assert data.zeta is not None, "Zeta must be initialized before adding constraints"
    assert data.in_horizon is not None, "In horizon must be initialized before adding constraints"
    # Separate and reshape waypoints into position, velocity, and acceleration vectors
    des_pos = data.waypoints["pos"][data.rank][data.in_horizon].flatten()
    des_vel = data.waypoints["vel"][data.rank][data.in_horizon].flatten()
    des_acc = data.waypoints["acc"][data.rank][data.in_horizon].flatten()

    # Extract penalized steps from first column of waypoints
    # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
    step_idx = data.t_discrete[data.in_horizon]
    t_idx = jp.repeat(step_idx, 3) * 3 + jp.tile(jp.arange(3, dtype=int), len(step_idx))

    x_0 = data.x_0[data.rank]
    linear_cost = data.linear_cost[data.rank]
    quad_cost = data.quad_cost[data.rank]
    linear_cost += data.linear_cost_smoothness_const @ x_0

    # Add constraints. See Ben Sprenger's master thesis for derivations
    q, Q, data = _add_waypoint_pos_constraint(data, settings, t_idx, des_pos, x_0)
    if q is not None and Q is not None:
        linear_cost, quad_cost = linear_cost + q, quad_cost + Q
    q, Q, data = _add_waypoint_vel_constraint(data, settings, t_idx, des_vel, x_0)
    if q is not None and Q is not None:
        linear_cost, quad_cost = linear_cost + q, quad_cost + Q
    q, Q, data = _add_waypoint_acc_constraint(data, settings, t_idx, des_acc, x_0)
    if q is not None and Q is not None:
        linear_cost, quad_cost = linear_cost + q, quad_cost + Q
    q, data = _add_input_continuity_constraint(data, settings, t_idx, x_0)
    if q is not None:
        linear_cost = linear_cost + q
    data = _add_pos_limit_constraint(data, settings, x_0)
    data = _add_vel_limit_constraint(data, settings, x_0)
    data = _add_acc_limit_constraint(data, settings, x_0)
    data = _add_collision_constraint(data, settings, x_0)
    data = data.replace(linear_cost=data.linear_cost.at[data.rank].set(linear_cost))
    data = data.replace(quad_cost=data.quad_cost.at[data.rank].set(quad_cost))
    return data


def _add_waypoint_pos_constraint(
    data: SolverData, settings: SolverSettings, t_idx: Array, des_pos: Array, x_0: Array
) -> tuple[Array | None, Array | None, SolverData]:
    G_wp = data.matrices.M_p_S_u_W_input[t_idx]
    h_wp = des_pos - data.matrices.M_p_S_x[t_idx] @ x_0
    q, Q = None, None
    if settings.pos_constraints == "hard":
        data = data.replace(
            pos_constraint=EqualityConstraint.init(G_wp, h_wp, settings.waypoints_pos_tol)
        )
    elif settings.pos_constraints == "soft":
        q = -2 * settings.pos_weight * G_wp.T @ h_wp
        Q = 2 * settings.pos_weight * G_wp.T @ G_wp
    return q, Q, data


def _add_waypoint_vel_constraint(
    data: SolverData, settings: SolverSettings, t_idx: Array, des_vel: Array, x_0: Array
) -> tuple[Array | None, Array | None, SolverData]:
    G_wv = data.matrices.M_v_S_u_W_input[t_idx]
    h_wv = des_vel - data.matrices.M_v_S_x[t_idx] @ x_0
    q, Q = None, None
    if settings.vel_constraints == "hard":
        constr = EqualityConstraint.init(G_wv, h_wv, settings.waypoints_vel_tol)
        data = data.replace(vel_constraint=constr)
    elif settings.vel_constraints == "soft":
        q = -2 * settings.vel_weight * G_wv.T @ h_wv
        Q = 2 * settings.vel_weight * G_wv.T @ G_wv
    return q, Q, data


def _add_waypoint_acc_constraint(
    data: SolverData, settings: SolverSettings, t_idx: Array, des_acc: Array, x_0: Array
) -> tuple[Array | None, Array | None, SolverData]:
    G_wa = data.matrices.M_a_S_u_prime_W_input[t_idx]
    h_wa = des_acc - data.matrices.M_a_S_x_prime[t_idx] @ x_0
    q, Q = None, None
    if settings.acc_constraints == "hard":
        constr = EqualityConstraint.init(G_wa, h_wa, settings.waypoints_acc_tol)
        data = data.replace(acc_constraint=constr)
    elif settings.acc_constraints == "soft":
        q = -2 * settings.acc_weight * G_wa.T @ h_wa
        Q = 2 * settings.acc_weight * G_wa.T @ G_wa
    return q, Q, data


def _add_input_continuity_constraint(
    data: SolverData, settings: SolverSettings, t_idx: Array, x_0: Array
) -> tuple[Array | None, SolverData]:
    u_0 = data.u_pos[data.rank, 0]
    u_dot_0 = data.u_vel[data.rank, 0]
    u_ddot_0 = data.u_acc[data.rank, 0]
    h_u = jp.concatenate([u_0, u_dot_0, u_ddot_0])
    q = None
    if settings.input_continuity_constraints == "hard":
        cnstr = EqualityConstraint.init(data.matrices.G_u, h_u, settings.input_continuity_tol)
        data = data.replace(input_continuity_constraint=cnstr)
    elif settings.input_continuity_constraints == "soft":
        q = -2 * settings.input_continuity_weight * data.matrices.G_u.T @ h_u
    return q, data


def _add_pos_limit_constraint(data: SolverData, settings: SolverSettings, x_0: Array) -> SolverData:
    upper = jp.tile(settings.pos_max, settings.K + 1) - data.matrices.M_p_S_x @ x_0
    lower = -jp.tile(settings.pos_min, settings.K + 1) + data.matrices.M_p_S_x @ x_0
    h_p = jp.concatenate([upper, lower])
    constr = InequalityConstraint.init(
        data.matrices.G_p, h_p, settings.pos_limit_tol, active_range=ACTIVE_RANGE
    )
    return data.replace(max_pos_constraint=constr)


def _add_vel_limit_constraint(data: SolverData, settings: SolverSettings, x_0: Array) -> SolverData:
    c_v = data.matrices.M_v_S_x @ x_0
    constr = PolarInequalityConstraint.init(
        data.matrices.M_v_S_u_W_input,
        c_v,
        upr_bound=settings.vel_max,
        tol=settings.vel_limit_tol,
        active_range=ACTIVE_RANGE,
    )
    return data.replace(max_vel_constraint=constr)


def _add_acc_limit_constraint(data: SolverData, settings: SolverSettings, x_0: Array) -> SolverData:
    c_a = data.matrices.M_a_S_x_prime @ x_0
    G = data.matrices.M_a_S_u_prime_W_input
    constr = PolarInequalityConstraint.init(
        G, c_a, upr_bound=settings.acc_max, tol=settings.acc_limit_tol, active_range=ACTIVE_RANGE
    )
    return data.replace(max_acc_constraint=constr)


def _add_collision_constraint(data: SolverData, settings: SolverSettings, x_0: Array) -> SolverData:
    n_collisions = min(settings.max_collisions, data.n_drones - 1)
    min_dist = jp.min(data.distance_matrix[data.rank], axis=-1)
    closest_drones = jp.argsort(min_dist)[:n_collisions]
    G_c_batched = jp.zeros((n_collisions, 3 * (settings.K + 1), 3 * (settings.N + 1)))
    c_c_batched = jp.zeros((n_collisions, 3 * (settings.K + 1)))
    # Add the closest n drones to the collision constraint. If there are less collisions, we
    # deactivate them with the active flag.
    envelope = jp.tile(1 / settings.collision_envelope, settings.K + 1)
    for i, d in enumerate(closest_drones):
        c_c = envelope * (data.matrices.M_p_S_x @ x_0 - data.pos[d].flatten())
        G_c_batched = G_c_batched.at[i].set(envelope[:, None] * data.matrices.M_p_S_u_W_input)
        c_c_batched = c_c_batched.at[i].set(c_c)
    mask = jp.zeros(n_collisions, dtype=bool)
    mask = mask.at[:n_collisions].set(min_dist[closest_drones] <= 1.0)
    constr = PolarInequalityConstraint.init(
        G_c_batched,
        c_c_batched,
        lwr_bound=1.0,
        tol=settings.collision_tol,
        mask=mask,
        active_range=ACTIVE_RANGE,
    )
    return data.replace(collision_constraints=constr)


def _spline2states(data: SolverData, settings: SolverSettings) -> SolverData:
    """Extract position, velocity, and acceleration trajectories from solution coefficients."""
    K = settings.K
    # Extract position trajectory from state trajectory
    zeta = data.zeta[data.rank]
    x_0 = data.x_0[data.rank]
    pos = (data.matrices.S_x @ x_0 + data.matrices.S_u_W_input @ zeta).T.reshape((K + 1, 6))[:, :3]
    u_pos = (data.matrices.W @ zeta).T.reshape((K, 3))
    u_vel = (data.matrices.W_dot @ zeta).T.reshape((K, 3))
    u_acc = (data.matrices.W_ddot @ zeta).T.reshape(K, 3)
    # Get input position, velocity and acceleration from spline coefficients
    data = data.replace(
        pos=data.pos.at[data.rank].set(pos),
        u_pos=data.u_pos.at[data.rank].set(u_pos),
        u_vel=data.u_vel.at[data.rank].set(u_vel),
        u_acc=data.u_acc.at[data.rank].set(u_acc),
    )
    return data


def _am_solve(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Conducts actual solving process implementing optimization algorithm.

    Not meant to be overridden by child classes.
    """
    rho = settings.rho_init
    zeta = data.zeta[data.rank]  # Previously was zero initialized, now uses previous solution
    bregman_mult = jp.zeros(data.quad_cost[data.rank].shape[0])  # Bregman multiplier
    # Update constraints with initial solution
    data = data.replace(
        max_pos_constraint=InequalityConstraint.update(data.max_pos_constraint, zeta),
        max_vel_constraint=PolarInequalityConstraint.update(data.max_vel_constraint, zeta),
        max_acc_constraint=PolarInequalityConstraint.update(data.max_acc_constraint, zeta),
        collision_constraints=PolarInequalityConstraint.update(data.collision_constraints, zeta),
    )
    # Aggregate quadratic and linear terms from all constraints
    Q_cnstr = _quadratic_constraint_costs(data)
    q_cnstr = _linear_constraint_costs(data)

    def cond_fn(
        val: tuple[int, Array, float, Array, Array, SolverData, SolverSettings, Array],
    ) -> bool:
        i, zeta, _, _, _, _, data, settings = val
        return (i < settings.max_iters) & ~_constraints_satisfied(zeta, data)

    def loop_fn(
        val: tuple[int, Array, float, Array, Array, SolverData, SolverSettings, Array],
    ) -> tuple[int, Array, SolverData]:
        # Unpack values
        i, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings = val
        # Solve QP
        Q = data.quad_cost[data.rank] + rho * Q_cnstr
        q = data.linear_cost[data.rank] + rho * (q_cnstr - bregman_mult)
        zeta = jp.linalg.solve(Q, -q)
        # Update constraints
        data = data.replace(
            max_pos_constraint=InequalityConstraint.update(data.max_pos_constraint, zeta),
            max_vel_constraint=PolarInequalityConstraint.update(data.max_vel_constraint, zeta),
            max_acc_constraint=PolarInequalityConstraint.update(data.max_acc_constraint, zeta),
            collision_constraints=PolarInequalityConstraint.update(
                data.collision_constraints, zeta
            ),
        )
        Q_cnstr = _quadratic_constraint_costs(data)
        q_cnstr = _linear_constraint_costs(data)
        # Calculate Bregman multiplier
        bregman_mult = bregman_mult - 0.5 * (Q_cnstr @ zeta + q_cnstr)
        # Increase penalty parameter. The factor 5.0 is a heuristic, other scheduling schemes might
        # improve performance.
        rho = jp.clip(rho * 5.0, max=settings.rho_max)
        return i + 1, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings

    # Compiled equivalent to
    # while cond_fn(...):
    #     loop_fn(...)
    i, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings = jax.lax.while_loop(
        cond_fn, loop_fn, (0, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings)
    )
    data = data.replace(zeta=data.zeta.at[data.rank].set(zeta))
    return i != settings.max_iters, i, data


def _quadratic_constraint_costs(data: SolverData) -> Array:
    Q_cnstr = jp.zeros_like(data.quad_cost[data.rank])
    if data.pos_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.pos_constraint)
    if data.vel_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.vel_constraint)
    if data.acc_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.acc_constraint)
    if data.input_continuity_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.input_continuity_constraint)
    Q_cnstr += InequalityConstraint.quadratic_term(data.max_pos_constraint)
    Q_cnstr += PolarInequalityConstraint.quadratic_term(data.max_vel_constraint)
    Q_cnstr += PolarInequalityConstraint.quadratic_term(data.max_acc_constraint)
    Q_cnstr += jp.sum(PolarInequalityConstraint.quadratic_term(data.collision_constraints), axis=0)
    return Q_cnstr


def _linear_constraint_costs(data: SolverData) -> Array:
    q_cnstr = jp.zeros_like(data.linear_cost[data.rank])
    if data.pos_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.pos_constraint)
    if data.vel_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.vel_constraint)
    if data.acc_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.acc_constraint)
    if data.input_continuity_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.input_continuity_constraint)
    q_cnstr += InequalityConstraint.linear_term(data.max_pos_constraint)
    q_cnstr += PolarInequalityConstraint.linear_term(data.max_vel_constraint)
    q_cnstr += PolarInequalityConstraint.linear_term(data.max_acc_constraint)
    q_cnstr += jp.sum(PolarInequalityConstraint.linear_term(data.collision_constraints), axis=0)
    return q_cnstr


def _constraints_satisfied(zeta: Array, data: SolverData) -> Array:
    """Check if all constraints are satisfied"""
    satisfied = jp.all(PolarInequalityConstraint.satisfied(data.collision_constraints, zeta))
    satisfied &= InequalityConstraint.satisfied(data.max_pos_constraint, zeta)
    satisfied &= PolarInequalityConstraint.satisfied(data.max_vel_constraint, zeta)
    satisfied &= PolarInequalityConstraint.satisfied(data.max_acc_constraint, zeta)
    if data.pos_constraint:
        satisfied &= EqualityConstraint.satisfied(data.pos_constraint, zeta)
    if data.vel_constraint:
        satisfied &= EqualityConstraint.satisfied(data.vel_constraint, zeta)
    if data.acc_constraint:
        satisfied &= EqualityConstraint.satisfied(data.acc_constraint, zeta)
    if data.input_continuity_constraint:
        satisfied &= EqualityConstraint.satisfied(data.input_continuity_constraint, zeta)
    return satisfied


@partial(jax.jit, static_argnums=(2, 3))
def _filter_horizon(times: Array, t: float, K: int, mpc_freq: float) -> tuple[int, int, Array]:
    """Extract waypoints in current horizon.

    Args:
        times: Waypoints times
        t: Current time
        K: Horizon length
        mpc_freq: MPC frequency

    Returns:
        A mask of waypoints within current horizon and the rounded times for all waypoints
    """
    assert isinstance(times, Array), f"Waypoints must be an Array, is {type(times)}"
    assert times.ndim == 1, "Waypoints must be a 2D array"
    # Round times to nearest discrete time step relative to current time
    rounded_times = jp.asarray(jp.round((times - t) * mpc_freq), dtype=int)
    # Find time steps with waypoints within current horizon
    in_horizon = (rounded_times > 0) & (rounded_times <= K)
    start = jp.argmax(in_horizon)
    end = len(in_horizon) - jp.argmax(jp.flip(in_horizon))
    return start, end, rounded_times
