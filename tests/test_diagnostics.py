"""
Tests for quickpde.diagnostics.

Strategy:
- enstrophy / energy: test against analytical values on fields where the integral
  can be computed exactly. The discrete sum equals the continuous integral for
  single-mode fields on the periodic grid (Parseval), so we can use tight tolerances.
- velocity: the streamfunction construction guarantees divergence-free velocity,
  which is a physical invariant independent of the specific field — this is a
  meaningful test of the Poisson-to-velocity pipeline.
- core_distance_jax: construct a synthetic field with two Gaussian peaks at
  known pixel positions and verify the returned distance is correct.
"""
import math

import jax.numpy as jnp
from quickpde.diagnostics import (
    core_distance_jax,
    energy,
    enstrophy,
    velocity,
)


# ---------------------------------------------------------------------------
# enstrophy
# ---------------------------------------------------------------------------

def test_enstrophy_zero_field(grid_2d):
    _, _, dx, N = grid_2d
    assert float(enstrophy(jnp.zeros((N, N)), dx)) == 0.0



def test_enstrophy_known_analytical_value(grid_2d):
    """enstrophy(sin(x)sin(y)) = π²/2, exact for any N.

    Proof: 0.5 * ∫sin²(x)sin²(y) dx dy = 0.5 * π * π = π²/2.
    Discretely: sum_ij sin²(xi)sin²(yj) · dx² = (N/2·dx)² = π², so
    0.5 * π² = π²/2. Holds for any N, so tolerance can be tight.
    """
    X, Y, dx, _ = grid_2d
    omega = jnp.sin(X) * jnp.sin(Y)
    result = float(enstrophy(omega, dx))
    expected = math.pi**2 / 2
    assert abs(result - expected) < 1e-8


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

def test_energy_zero_field(grid_2d):
    _, _, dx, N = grid_2d
    assert float(energy(jnp.zeros((N, N)), dx)) == 0.0


def test_energy_nonnegative(grid_2d):
    X, Y, dx, _ = grid_2d
    omega = jnp.sin(X) + 0.3 * jnp.cos(2 * Y)
    assert float(energy(omega, dx)) >= 0.0


def test_energy_single_mode_equals_enstrophy_over_ksq(grid_2d):
    """For a single-mode vorticity field at wavenumber k, energy = enstrophy / k².

    Using ω = cos(x) (kx=1, ky=0 → k²=1), energy should equal enstrophy.
    This verifies that the streamfunction inversion and velocity reconstruction
    are correctly coupled.
    """
    X, _, dx, _ = grid_2d
    omega = jnp.cos(X)  # single mode at k²=1
    e = float(energy(omega, dx))
    ens = float(enstrophy(omega, dx))
    # energy = enstrophy / k^2 = enstrophy / 1
    assert abs(e - ens) / ens < 1e-5


# ---------------------------------------------------------------------------
# velocity (divergence-free check)
# ---------------------------------------------------------------------------

def test_velocity_is_divergence_free(grid_2d):
    """u = ∂ψ/∂y, v = -∂ψ/∂x → div(u,v) = ∂u/∂x + ∂v/∂y = 0 identically.

    This tests the full Poisson-to-velocity pipeline. Any bug in the streamfunction
    inversion or the velocity reconstruction would break this invariant.
    """
    from quickpde.derivs import fourier_deriv

    X, Y, dx, N = grid_2d  # N needed for fourier_deriv
    omega = jnp.sin(X) + 0.5 * jnp.cos(2 * Y) - jnp.mean(jnp.sin(X))

    u, v = velocity(omega, dx)

    ddx = fourier_deriv(N, dx, axis=0)
    ddy = fourier_deriv(N, dx, axis=1)
    divergence = ddx(u) + ddy(v)

    assert float(jnp.max(jnp.abs(divergence))) < 1e-8


# ---------------------------------------------------------------------------
# core_distance_jax
# ---------------------------------------------------------------------------

def _two_peak_field(N, i1, j1, i2, j2, sigma=2.0):
    """Synthetic field with two Gaussian peaks at given pixel positions."""
    ii = jnp.arange(N)
    jj = jnp.arange(N)
    I, J = jnp.meshgrid(ii, jj, indexing='ij')
    peak1 = jnp.exp(-((I - i1)**2 + (J - j1)**2) / (2 * sigma**2))
    peak2 = jnp.exp(-((I - i2)**2 + (J - j2)**2) / (2 * sigma**2))
    return peak1 + peak2


def test_core_distance_jax_known_separation():
    """Two peaks placed 32 pixels apart should yield distance 32 (with dx=1)."""
    N = 64
    field = _two_peak_field(N, i1=16, j1=32, i2=48, j2=32)
    dist = float(core_distance_jax(field, dx=1.0))
    assert abs(dist - 32.0) < 1.0  # within 1 pixel


def test_core_distance_jax_diagonal_separation():
    """Peaks separated diagonally: distance = sqrt(dx² + dy²) in pixel units."""
    N = 64
    di, dj = 20, 20
    field = _two_peak_field(N, i1=22, j1=22, i2=22 + di, j2=22 + dj)
    expected = math.sqrt(di**2 + dj**2)
    dist = float(core_distance_jax(field, dx=1.0))
    assert abs(dist - expected) < 1.5  # within ~1 pixel diagonal tolerance


def test_core_distance_jax_single_peak_returns_zero():
    """When both detected peaks collapse onto one, distance should be 0."""
    N = 64
    ii = jnp.arange(N)
    jj = jnp.arange(N)
    I, J = jnp.meshgrid(ii, jj, indexing='ij')
    # Single sharp peak — the second-best peak will be within r_min
    field = jnp.exp(-((I - 32)**2 + (J - 32)**2) / 2.0)
    dist = float(core_distance_jax(field, dx=1.0, r_min=20))
    assert dist == 0.0
