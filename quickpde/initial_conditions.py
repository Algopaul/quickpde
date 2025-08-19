import jax.numpy as jnp

from quickpde.config import Config
from quickpde.grid import get_grid


def get_initial_condition(cfg: Config):
  pde_dict = {
      'rotation_2d': rotation_2d,
  }
  return pde_dict[cfg.pde](cfg)


def rotation_2d(cfg: Config):
  (X, Y), _ = get_grid(cfg)
  return jnp.exp(-cfg.ic_sharpness * ((X - 1.0)**2 + Y**2))
