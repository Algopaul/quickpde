"""
Tests for PDE classes in quickpde.pdes.

Strategy:
- Registry: verify dispatch and error handling.
- RHS correctness: test the RHS on inputs with known outputs (e.g. constant field
  under rotation has zero tendency), not full trajectories.
- Conservation: run exactly 1 step with a tiny grid and verify a conserved
  quantity holds. Rotation is norm-preserving; this catches RHS assembly bugs.
- Config wiring: verify that a config field (hyperviscosity_mag) actually reaches
  the computation. This is the class of bug that existed before the refactor.
"""
import math

import jax.numpy as jnp
import jax.random as jrd
import pytest

from quickpde.config.base import Config, VorticityConfig
from quickpde.pdes import PDE, Rotation2d, ShallowWater, Vorticity, Wave


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pde_name,expected_cls', [
    ('rotation_2d', Rotation2d),
    ('wave2d', Wave),
    ('swe2d', ShallowWater),
])
def test_from_config_dispatch(pde_name, expected_cls):
    cfg = Config(
        pde=pde_name,
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        ic_sharpness=2.0,
    )
    assert isinstance(PDE.from_config(cfg), expected_cls)


def test_from_config_unknown_raises():
    cfg = Config(pde='does_not_exist')
    with pytest.raises(ValueError, match='does_not_exist'):
        PDE.from_config(cfg)


# ---------------------------------------------------------------------------
# Rotation2d RHS
# ---------------------------------------------------------------------------

def test_rotation_rhs_constant_field_is_zero(rotation_cfg):
    """A spatially uniform field has zero tendency under rotation.

    RHS = X·∂u/∂y - Y·∂u/∂x. For constant u, both derivatives are zero.
    This tests that the operator is assembled correctly and doesn't accidentally
    multiply by the grid coordinates twice.
    """
    pde = Rotation2d(rotation_cfg)
    u = jnp.ones((rotation_cfg.axis_points, rotation_cfg.axis_points))
    dudt = pde.rhs(u)
    assert float(jnp.max(jnp.abs(dudt))) < 1e-10


def test_rotation_rhs_output_shape(rotation_cfg):
    pde = Rotation2d(rotation_cfg)
    u = pde.initial_condition(rotation_cfg)
    assert pde.rhs(u).shape == u.shape


def test_rotation_rhs_output_finite(rotation_cfg):
    pde = Rotation2d(rotation_cfg)
    u = pde.initial_condition(rotation_cfg)
    assert jnp.isfinite(pde.rhs(u)).all()


def test_rotation_one_step_norm_preserved(rotation_cfg):
    """Rotation is an isometry: ||u||² should not change under the flow.

    The continuous operator is antisymmetric, so RK4 should preserve the L2
    norm to O(dt^5). With dt=1e-3 this is < 1e-12.
    A bug in the RHS sign or coefficient would show up as a clear norm change.
    """
    pde = PDE.from_config(rotation_cfg)
    traj, _ = pde.solve(rotation_cfg)
    ic = pde.initial_condition(rotation_cfg)
    norm_before = float(jnp.sum(ic**2))
    norm_after = float(jnp.sum(traj[0]**2))
    assert abs(norm_after - norm_before) / norm_before < 1e-8


# ---------------------------------------------------------------------------
# Wave RHS
# ---------------------------------------------------------------------------

def test_wave_rhs_output_shape(wave_cfg):
    from quickpde.pdes import Wave
    pde = Wave(wave_cfg)
    u = pde.initial_condition(wave_cfg)
    assert pde.rhs(u).shape == u.shape


def test_wave_rhs_output_finite(wave_cfg):
    from quickpde.pdes import Wave
    pde = Wave(wave_cfg)
    u = pde.initial_condition(wave_cfg)
    assert jnp.isfinite(pde.rhs(u)).all()


# ---------------------------------------------------------------------------
# Vorticity: config wiring
# ---------------------------------------------------------------------------

def test_vorticity_hyperviscosity_mag_wiring():
    """cfg.vorticity.hyperviscosity_mag must reach get_step, not be hardcoded.

    Two instances with vastly different hyperviscosity_mag should produce
    different outputs after one step on a field with high-frequency content.
    """
    N = 16
    cfg_weak = Config(
        pde='vorticity',
        axis_points=N,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        viscosity=0.0,
        vorticity=VorticityConfig(hyperviscosity_mag=1.0),
    )
    cfg_strong = Config(
        pde='vorticity',
        axis_points=N,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        viscosity=0.0,
        vorticity=VorticityConfig(hyperviscosity_mag=1e12),
    )

    # Random field: energy spread across all wavenumbers, including high modes
    # that are maximally damped by hyperviscosity. sin(4x)=0 everywhere on N=8
    # grid, so we use a random IC to avoid degenerate zero fields.
    x0 = jrd.normal(jrd.key(0), (N, N))

    step_weak = Vorticity().get_step(cfg_weak)
    step_strong = Vorticity().get_step(cfg_strong)

    out_weak = step_weak(0, x0)
    out_strong = step_strong(0, x0)

    assert not jnp.allclose(out_weak, out_strong), (
        "hyperviscosity_mag had no effect — likely still hardcoded"
    )
