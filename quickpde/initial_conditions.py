import jax
import jax.numpy as jnp

from quickpde.config import Config
from quickpde.grid import get_grid


def get_initial_condition(cfg: Config):
  pde_dict = {
      'rotation_2d': rotation_2d,
      'rde_1d': rde_1d,
  }
  return pde_dict[cfg.pde](cfg)


def rotation_2d(cfg: Config):
  (X, Y), _ = get_grid(cfg)
  return jnp.exp(-cfg.ic_sharpness * ((X - 1.0)**2 + Y**2))


def rde_1d(cfg: Config):
  x, _ = get_grid(cfg)
  assert isinstance(x, jax.Array)
  # eta_0 = jnp.exp(-cfg.ic_sharpness * (x - 1.0)**2)
  eta_0 = 3 / 2 * (1.0 / jnp.cosh(x - 1.0))**2
  lam_0 = jnp.zeros_like(x)
  return jnp.hstack((eta_0, lam_0))
