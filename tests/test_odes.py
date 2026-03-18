"""
Tests for ODE time integration in quickpde.odesolve.

Strategy:
- get_rk: apply to a scalar ODE with a known exact solution (exponential decay).
  RK4 local error is O(dt^5), so with dt=0.1 the error should be < 1e-6.
- get_ode_solver: verify output shapes and timepoint values without running any
  expensive simulation (tiny grid, 1 stored step).
"""
import math

import jax.numpy as jnp
from quickpde.config.base import Config
from quickpde.odesolve import get_rk


def test_rk4_scalar_exponential_decay():
    """dy/dt = -y, y(0)=1 → y(dt) = e^{-dt}. Checks RK4 is correctly assembled."""
    dt = 0.1
    step = get_rk(lambda x: -x, dt)
    y1 = step(0, jnp.array(1.0))
    expected = math.exp(-dt)
    # RK4 local error is O(dt^5); with dt=0.1 expect < 1e-6
    assert abs(float(y1) - expected) < 1e-6


def test_rk4_preserves_shape():
    """Step function must return an array with the same shape as the input."""
    dt = 0.01
    step = get_rk(lambda x: -x, dt)
    field = jnp.ones((8, 8))
    out = step(0, field)
    assert out.shape == field.shape


def test_ode_solver_trajectory_shape():
    """Trajectory has shape (n_stored, *field_shape)."""
    from quickpde.pdes import PDE

    cfg = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=3e-3,   # 3 timesteps stored
        store_every=1,
        ic_sharpness=5.0,
    )
    pde = PDE.from_config(cfg)
    traj, _ = pde.solve(cfg)
    assert traj.shape == (3, 8, 8)


def test_ode_solver_store_every():
    """store_every=2 halves the number of stored frames."""
    from quickpde.pdes import PDE

    cfg_all = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=4e-3,
        store_every=1,
        ic_sharpness=5.0,
    )
    cfg_every2 = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=4e-3,
        store_every=2,
        ic_sharpness=5.0,
    )
    traj_all, _ = PDE.from_config(cfg_all).solve(cfg_all)
    traj_every2, _ = PDE.from_config(cfg_every2).solve(cfg_every2)
    assert traj_all.shape[0] == 2 * traj_every2.shape[0]


def test_ode_solver_timepoints_are_endpoint_times():
    """Returned timepoints should be the END time of each stored interval."""
    from quickpde.pdes import PDE

    dt = 1e-3
    store_every = 2
    cfg = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=dt,
        t_end=4e-3,
        store_every=store_every,
        ic_sharpness=5.0,
    )
    _, timepoints = PDE.from_config(cfg).solve(cfg)
    # With t_end=4e-3, dt=1e-3, store_every=2: stored at t=2e-3 and t=4e-3
    expected = jnp.array([2e-3, 4e-3])
    assert jnp.allclose(timepoints, expected, atol=1e-10)
