from __future__ import annotations

from flax.struct import dataclass, field
from numpy.typing import NDArray


@dataclass
class SolverSettings:
    # AMSwarm solver iteration settings
    max_iters: int = field(pytree_node=False)  # Maximum number of iterations
    rho_init: float  # Initial value of rho
    rho_max: float  # Maximum allowable value of rho

    # Constraints
    pos_constraints: bool = field(pytree_node=False)
    vel_constraints: bool = field(pytree_node=False)
    acc_constraints: bool = field(pytree_node=False)
    input_continuity_constraints: bool = field(pytree_node=False)
    max_collisions: int = field(pytree_node=False)

    # Weights
    pos_weight: float
    vel_weight: float
    acc_weight: float
    smoothness_weight: float
    input_smoothness_weight: float
    input_continuity_weight: float

    # Limits
    pos_min: NDArray
    pos_max: NDArray
    vel_max: float
    acc_max: float
    collision_envelope: NDArray

    # MPC
    K: int = field(pytree_node=False)
    N: int = field(pytree_node=False)
    freq: float = field(pytree_node=False)
    bf_gamma: float
    waypoints_pos_tol: float
    waypoints_vel_tol: float
    waypoints_acc_tol: float
    pos_limit_tol: float
    vel_limit_tol: float
    acc_limit_tol: float
    input_continuity_tol: float
    collision_tol: float
