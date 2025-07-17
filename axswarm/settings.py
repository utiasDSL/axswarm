from __future__ import annotations

from typing import Literal

from flax.struct import dataclass, field
from numpy.typing import NDArray


@dataclass
class SolverSettings:
    # AMSwarm solver iteration settings
    max_iters: int = field(pytree_node=False)  # Maximum number of iterations
    rho_init: float  # Initial value of rho
    rho_max: float  # Maximum allowable value of rho

    # Constraints
    pos_constraints: Literal["hard", "soft", False] = field(pytree_node=False)
    vel_constraints: Literal["hard", "soft", False] = field(pytree_node=False)
    acc_constraints: Literal["hard", "soft", False] = field(pytree_node=False)
    input_continuity_constraints: Literal["hard", "soft", False] = field(pytree_node=False)
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

    def __post_init__(self):
        constraint_types = ["hard", "soft", False]
        if self.pos_constraints not in constraint_types:
            raise ValueError(f"Invalid setting for pos_constraints: {self.pos_constraints}")
        if self.vel_constraints not in constraint_types:
            raise ValueError(f"Invalid setting for vel_constraints: {self.vel_constraints}")
        if self.acc_constraints not in constraint_types:
            raise ValueError(f"Invalid setting for acc_constraints: {self.acc_constraints}")
        if self.input_continuity_constraints not in constraint_types:
            raise ValueError(
                "Invalid setting for input_continuity_constraints: "
                f"{self.input_continuity_constraints}"
            )
