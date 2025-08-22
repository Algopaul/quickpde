import jax
import jax.numpy as jnp
import jax.random as jrd

from quickpde.config import Config
from quickpde.grid import get_grid


def power_spectrum(kx, ky, alpha=2.0):
  """Power spectrum for the Gaussian random field."""
  k = jnp.sqrt(kx**2 + ky**2)
  return jnp.where(jnp.abs(k) <= 1e-12, 0.0, k**(-alpha))


def gaussian_random_field(
    magnitude,
    size0,
    size1,
    alpha=2.0,
    random_seed: int | None = None,
):
  if random_seed is None:
    random_seed = 0
  random_key = jrd.key(random_seed)
  nx, ny = size0, size1
  kx = jnp.fft.fftfreq(nx).reshape(-1, 1)
  ky = jnp.fft.fftfreq(ny).reshape(1, -1)
  amplitude = jnp.sqrt(power_spectrum(kx, ky, alpha))
  k0, k1 = jrd.split(random_key)

  phase = jrd.normal(k0, (nx, ny)) + 1j * jrd.normal(k1, (nx, ny))
  fft_field = amplitude * phase

  field = jnp.fft.ifft2(fft_field).real
  var = jnp.var(field.reshape(-1))
  return magnitude * field / jnp.sqrt(var)


def init_field(cfg: Config):
  n = cfg.axis_points
  return gaussian_random_field(1.0, n, n, 2.0)


def gaussian_bump_2d(X, Y, bump_pos, cfg: Config):
  Xs = X - bump_pos[0]
  Ys = Y - bump_pos[1]
  return jnp.exp(-cfg.ic_sharpness * (Xs**2 + Ys**2))


def init_bump(cfg: Config):
  (X, Y), _ = get_grid(cfg)
  return gaussian_bump_2d(X, Y, [-1.0, 0], cfg)


def double_bump(cfg: Config):
  (X, Y), _ = get_grid(cfg)
  r = cfg.vorticity.bump_distance / 2
  a = cfg.vorticity.bump_angle
  p1 = [r * jnp.cos(a), r * jnp.sin(a)]
  p2 = [-r * jnp.cos(a), -r * jnp.sin(a)]
  b1 = gaussian_bump_2d(X, Y, p1, cfg)
  b2 = gaussian_bump_2d(X, Y, p2, cfg)
  b = b1 + b2
  return b - jnp.mean(b)
