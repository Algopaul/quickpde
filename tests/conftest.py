import math

import jax
import jax.numpy as jnp
import pytest

# Enable 64-bit throughout the test suite so that spectral methods and
# conservation tests can use tight tolerances. Must happen before any JAX
# compilation (i.e. at module import time, before any test runs).
jax.config.update('jax_enable_x64', True)

from quickpde.config.base import Config, VorticityConfig


@pytest.fixture
def grid_1d():
    """1D periodic grid on [0, 2π) with N=64 points."""
    N = 64
    x = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    dx = float(x[1] - x[0])
    return x, dx, N


@pytest.fixture
def grid_2d():
    """2D periodic meshgrid on [0, 2π)² with N=64 points per axis."""
    N = 64
    x = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    dx = float(x[1] - x[0])
    X, Y = jnp.meshgrid(x, x, indexing='ij')
    return X, Y, dx, N


@pytest.fixture
def rotation_cfg():
    return Config(
        pde='rotation_2d',
        axis_points=16,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        store_every=1,
        ic_sharpness=5.0,
    )


@pytest.fixture
def wave_cfg():
    return Config(
        pde='wave2d',
        axis_points=16,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        store_every=1,
        ic_sharpness=5.0,
    )


@pytest.fixture
def rde_cfg():
    return Config(
        pde='rde_1d',
        domain_dim=1,
        axis_points=32,
        bound_x=(0.0, 2 * math.pi),
        dt=1e-3,
        t_end=1e-3,
        store_every=1,
        viscosity=1e-4,
        injection_rate=3.0,
    )
