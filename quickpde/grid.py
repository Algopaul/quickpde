import jax.numpy as jnp

from quickpde.config import Config


def get_grid(cfg: Config):
  a, b = cfg.bound_x
  x = jnp.linspace(a, b, cfg.axis_points, endpoint=False)
  stepsize = x[1] - x[0]
  X, Y = jnp.meshgrid(x, x, indexing='ij')
  return (X, Y), stepsize
