# test_derivs.py
import jax
import jax.numpy as jnp
import pytest

from quickpde.derivs import central_deriv, fourier_deriv

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def grid_1d():
  N = 128
  L = 2.0
  dx = L / N
  x = jnp.arange(N) * dx
  return x, dx, N, L


def test_fourier_deriv_sine(grid_1d):
  x, dx, N, L = grid_1d
  m = 3
  u = jnp.sin(2 * jnp.pi * m * x / L)
  dudx_true = (2 * jnp.pi * m / L) * jnp.cos(2 * jnp.pi * m * x / L)

  d_fourier = fourier_deriv(N, dx, axis=0, order=1)
  dudx_spec = d_fourier(u)

  assert dudx_spec.shape == u.shape
  assert jnp.max(jnp.abs(dudx_spec - dudx_true)) < 1e-10


def test_fourier_second_deriv_cos(grid_1d):
  x, dx, N, L = grid_1d
  m = 5
  k = 2 * jnp.pi * m / L
  u = jnp.cos(k * x)
  d2_true = -(k**2) * jnp.cos(k * x)

  d2_fourier = fourier_deriv(N, dx, axis=0, order=2)
  d2u_spec = d2_fourier(u)

  assert jnp.max(jnp.abs(d2u_spec - d2_true)) < 1e-10


def test_broadcasting_axis_2d():
  Nx, Ny = 64, 96
  Lx, Ly = 1.0, 2.0
  dx, dy = Lx / Nx, Ly / Ny
  x = jnp.arange(Nx) * dx
  y = jnp.arange(Ny) * dy
  X, Y = jnp.meshgrid(x, y, indexing="ij")

  u = jnp.sin(2 * jnp.pi * X) * jnp.cos(4 * jnp.pi * Y / Ly)
  dudx_true = (2 * jnp.pi) * jnp.cos(2 * jnp.pi * X) * jnp.cos(
      4 * jnp.pi * Y / Ly)
  dudy_true = (-4 * jnp.pi / Ly) * jnp.sin(2 * jnp.pi * X) * jnp.sin(
      4 * jnp.pi * Y / Ly)

  d_dx = fourier_deriv(Nx, dx, axis=0, order=1)
  d_dy = fourier_deriv(Ny, dy, axis=1, order=1)

  assert jnp.max(jnp.abs(d_dx(u) - dudx_true)) < 1e-10
  assert jnp.max(jnp.abs(d_dy(u) - dudy_true)) < 1e-10


def test_mismatch_raises():
  N = 32
  dx = 1.0 / N
  d_dx = fourier_deriv(N, dx, axis=0, order=1)
  u_bad = jnp.zeros((N + 1,))
  with pytest.raises(ValueError):
    _ = d_dx(u_bad)
