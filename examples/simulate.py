"""axswarm simulation of a spiral formation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np
import yaml
from crazyflow import Sim
from crazyflow.utils import enable_cache
from utils import draw_line, draw_points

from axswarm import SolverData, SolverSettings, solve

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

enable_cache()

logger = logging.getLogger(__name__)


np.random.seed(0)
rgbas = np.random.rand(5, 4)
rgbas[..., 3] = 1.0


def render_solutions(sim, trajectories: list[np.ndarray]):
    for i, trajectory in enumerate(trajectories):
        draw_points(sim, trajectory, rgba=rgbas[i], size=0.01)
        draw_line(sim, trajectory, rgba=rgbas[i])


def generate_waypoints(n_drones: int, n_points: int = 4, duration_sec: float = 10.0):
    """Waypoints have the following shape: [T, n_drones, 3]."""
    radius = 0.75
    phase = np.linspace(0, 2 * (1 - 1 / n_drones) * np.pi, n_drones)[..., None]
    t = np.tile(np.linspace(0, duration_sec, n_points), (n_drones, 1))
    x = np.cos(0.1 * t * np.pi + phase) * radius
    y = np.sin(0.1 * t * np.pi + phase) * radius
    z = np.ones_like(x) * np.linspace(0.5, 1.5, n_points)
    pos = np.stack([x, y, z], axis=-1)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    assert pos.shape == (n_drones, n_points, 3), f"Shape {pos.shape} != ({n_drones}, {n_points}, 3)"
    return {"time": t, "pos": pos, "vel": vel, "acc": acc}


def simulate_amswarmpy(sim, waypoints, render=False) -> NDArray:
    """Run the AMSwarmPy simulation.

    Args:
        sim: Simulation object containing parameters
        waypoints: Dictionary of waypoints for each drone

    Returns:
        Dictionary containing trajectory positions
    """
    with open(Path(__file__).resolve().parents[1] / "params/settings.yaml") as f:
        config = yaml.safe_load(f)
    settings = config["SolverSettings"]

    # Convert lists to numpy arrays
    for k, v in settings.items():
        if isinstance(v, list):
            settings[k] = np.asarray(v)
    settings = SolverSettings(**settings)

    # Setup simulation parameters
    n_drones = sim.n_drones
    n_steps = int(waypoints["time"][0, -1] * settings.freq)

    dynamics = config["Dynamics"]
    A, B = np.asarray(dynamics["A"]), np.asarray(dynamics["B"])
    A_prime, B_prime = np.asarray(dynamics["A_prime"]), np.asarray(dynamics["B_prime"])
    trajectories = np.zeros((n_steps, n_drones, 3))  # Initialize trajectories storage
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=settings.K,
        N=settings.N,
        A=A,
        B=B,
        A_prime=A_prime,
        B_prime=B_prime,
        freq=settings.freq,
        smoothness_weight=settings.smoothness_weight,
        input_smoothness_weight=settings.input_smoothness_weight,
        input_continuity_weight=settings.input_continuity_weight,
    )

    sim.reset()
    # Set initial position states to first waypoint for each drone
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(waypoints["pos"][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))

    for step in range(n_steps):
        t = step / settings.freq

        # states = np.concat((solver_data.u_pos[:, 0], solver_data.u_vel[:, 0]), axis=-1)
        pos, vel = (np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0]))
        states = np.concat((pos, vel), axis=-1)
        success, _, solver_data = solve(states, t, solver_data, settings)
        if not all(success):
            logger.warning("Solve failed")

        solver_data = solver_data.step(solver_data)
        # Switch to u_pos okay?
        # control[0, :, :3] = solver_data.pos[:, 1]
        control[0, :, :3] = solver_data.u_pos[:, 1]
        control[0, :, 3:6] = solver_data.u_vel[:, 1]

        sim.state_control(control)
        sim.step(sim.freq // settings.freq)
        if render:
            render_solutions(sim, solver_data.pos)
            for i in range(n_drones):
                draw_points(sim, waypoints["pos"][i], rgba=rgbas[i], size=0.02)
            sim.render()

        trajectories[step] = sim.data.states.pos[0]

    return trajectories


def plot_trajectories(sim, waypoints, pos):
    """Plot comparison of trajectories between AMSwarm and AMSwarmPy implementations."""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each drone's trajectory
    for i in range(sim.n_drones):
        # Plot AMSwarm trajectory
        p = pos[:, i, :]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=f"AMSwarm Drone {i}", color=rgbas[i])
        # Plot waypoints
        p = waypoints["pos"][i]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="x", color=rgbas[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Trajectories Comparison")
    ax.legend()
    plt.show()


def main(render: bool = False):
    sim = Sim(n_drones=5, freq=400, state_freq=80, attitude_freq=400, control="state")
    n_points = 7
    waypoints = generate_waypoints(sim.n_drones, n_points=n_points)

    results = simulate_amswarmpy(sim, waypoints, render=render)
    results = None
    tstart = time.perf_counter()
    results = simulate_amswarmpy(sim, waypoints, render=render)
    tstop = time.perf_counter()
    print(f"amswarm solve time: {tstop - tstart:.2f} s")
    sim.close()

    plot_trajectories(sim, waypoints, results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger.setLevel(logging.ERROR)
    fire.Fire(main)
