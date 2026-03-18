"""
Tests for derivative operators in quickpde.derivs.

Strategy:
- fourier_deriv: spectral methods are exact to machine precision for band-limited
  functions, so we test against analytical derivatives with tight tolerance.
- central_deriv / upwind_deriv: finite-difference methods have a specific
  convergence order. We test both (a) that the value is approximately correct
  and (b) that the error scales with the right power of h.
- poisson_solver: roundtrip test — apply Laplacian to the result, recover input.
"""
import math

import jax.numpy as jnp
from quickpde.derivs import (
    central_deriv,
    fourier_deriv,
    periodic_poisson_solver,
    upwind_deriv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_err(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def _fd_error(N, make_deriv):
    """Error of make_deriv(N, dx) applied to sin(x) vs cos(x)."""
    x = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    dx = float(x[1] - x[0])
    return _max_err(make_deriv(N, dx)(jnp.sin(x)), jnp.cos(x))


# ---------------------------------------------------------------------------
# fourier_deriv
# ---------------------------------------------------------------------------

def test_fourier_deriv_sin_cos(grid_1d):
    """d/dx sin(x) = cos(x) to machine precision on a periodic grid."""
    x, dx, N = grid_1d
    ddx = fourier_deriv(N, dx, axis=0)
    assert _max_err(ddx(jnp.sin(x)), jnp.cos(x)) < 1e-10


def test_fourier_deriv_wavenumber_scaling(grid_1d):
    """d/dx sin(kx) = k·cos(kx) — tests that the wavenumber factor is applied correctly."""
    x, dx, N = grid_1d
    k = 3
    ddx = fourier_deriv(N, dx, axis=0)
    assert _max_err(ddx(jnp.sin(k * x)), k * jnp.cos(k * x)) < 1e-9


def test_fourier_deriv_second_order(grid_1d):
    """d²/dx² sin(x) = -sin(x) to machine precision."""
    x, dx, N = grid_1d
    d2dx2 = fourier_deriv(N, dx, axis=0, order=2)
    assert _max_err(d2dx2(jnp.sin(x)), -jnp.sin(x)) < 1e-10


def test_fourier_deriv_2d_correct_axis(grid_2d):
    """On a 2D field f(x,y) = sin(y), d/dy should give cos(y); d/dx should give 0."""
    _, Y, dx, N = grid_2d
    field = jnp.sin(Y)
    ddy = fourier_deriv(N, dx, axis=1)
    ddx = fourier_deriv(N, dx, axis=0)
    assert _max_err(ddy(field), jnp.cos(Y)) < 1e-10
    assert _max_err(ddx(field), jnp.zeros_like(field)) < 1e-10


# ---------------------------------------------------------------------------
# central_deriv (order=1)
# ---------------------------------------------------------------------------

def test_central_deriv_value(grid_1d):
    """d/dx sin(x) ≈ cos(x) — checks the formula, not just shape."""
    x, dx, _ = grid_1d
    ddx = central_deriv(dx, axis=0)
    # Central difference error is O(h²); at N=64, h≈0.098, so error ≈ h²/6 ≈ 0.002
    assert _max_err(ddx(jnp.sin(x)), jnp.cos(x)) < 1e-2


def test_central_deriv_second_order_convergence():
    """Halving h should quarter the error (2nd-order convergence)."""
    make = lambda _N, dx: central_deriv(dx, axis=0)
    err32 = _fd_error(32, make)
    err64 = _fd_error(64, make)
    ratio = err32 / err64
    assert 3.5 < ratio < 4.5, f"Expected ~4, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# central_deriv (order=2)
# ---------------------------------------------------------------------------

def test_central_deriv_order2_value(grid_1d):
    """d²/dx² sin(x) ≈ -sin(x)."""
    x, dx, _ = grid_1d
    d2dx2 = central_deriv(dx, axis=0, order=2)
    assert _max_err(d2dx2(jnp.sin(x)), -jnp.sin(x)) < 1e-3


def test_central_deriv_order2_second_order_convergence():
    """Second derivative central difference is also 2nd-order in h."""
    x32 = jnp.linspace(0, 2 * math.pi, 32, endpoint=False)
    x64 = jnp.linspace(0, 2 * math.pi, 64, endpoint=False)
    dx32 = float(x32[1] - x32[0])
    dx64 = float(x64[1] - x64[0])

    err32 = _max_err(central_deriv(dx32, axis=0, order=2)(jnp.sin(x32)), -jnp.sin(x32))
    err64 = _max_err(central_deriv(dx64, axis=0, order=2)(jnp.sin(x64)), -jnp.sin(x64))
    ratio = err32 / err64
    assert 3.5 < ratio < 4.5, f"Expected ~4, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# upwind_deriv
# ---------------------------------------------------------------------------

def test_upwind_deriv_value(grid_1d):
    """d/dx sin(x) ≈ cos(x) — check the formula gives a reasonable result."""
    x, dx, _ = grid_1d
    ddx = upwind_deriv(dx, axis=0)
    # 1st-order upwind is less accurate; at N=64 error should still be < 0.1
    assert _max_err(ddx(jnp.sin(x)), jnp.cos(x)) < 0.1


def test_upwind_deriv_first_order_convergence():
    """Halving h should halve the error (1st-order convergence)."""
    make = lambda _N, dx: upwind_deriv(dx, axis=0)
    err32 = _fd_error(32, make)
    err64 = _fd_error(64, make)
    ratio = err32 / err64
    assert 1.7 < ratio < 2.3, f"Expected ~2, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# periodic_poisson_solver
# ---------------------------------------------------------------------------

def test_poisson_solver_roundtrip(grid_2d):
    """∇²(solve(ω)) ≈ ω for a smooth multi-mode field (spectral accuracy)."""
    X, Y, dx, N = grid_2d
    omega = jnp.sin(X) + jnp.sin(2 * Y) + 0.5 * jnp.sin(X) * jnp.cos(Y)
    # Remove mean (Poisson solution is only unique up to a constant; zero-mean ω is well-posed)
    omega = omega - jnp.mean(omega)

    solver = periodic_poisson_solver(N, dx)
    psi = solver(omega)

    # Verify ∇²ψ ≈ ω using the spectral Laplacian
    d2dx2 = fourier_deriv(N, dx, axis=0, order=2)
    d2dy2 = fourier_deriv(N, dx, axis=1, order=2)
    laplacian_psi = d2dx2(psi) + d2dy2(psi)

    assert _max_err(laplacian_psi, omega) < 1e-5
