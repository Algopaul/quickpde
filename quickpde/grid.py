import jax.numpy as jnp

from quickpde.config import Config


def get_grid(cfg: Config):
  if cfg.domain_dim == 1:
    a, b = cfg.bound_x
    x = jnp.linspace(a, b, cfg.axis_points, endpoint=False)
    stepsize = x[1] - x[0]
    return x, stepsize
  elif cfg.domain_dim == 2:
    a, b = cfg.bound_x
    x = jnp.linspace(a, b, cfg.axis_points, endpoint=False)
    stepsize = x[1] - x[0]
    X, Y = jnp.meshgrid(x, x, indexing='ij')
    return (X, Y), stepsize
  else:
    raise NotImplemented('Only 1d and 2d domains implemented.')
