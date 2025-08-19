import jax.numpy as jnp

from quickpde import derivs
from quickpde.config import Config
from quickpde.grid import get_grid


def get_pde(cfg: Config):
  pde_dict = {
      'rotation_2d': rotation_2d,
  }
  return pde_dict[cfg.pde](cfg)


def rotation_2d(cfg: Config):
  """Transport in 2d that just rotates the field

  Args:
    cfg (Config): Config struct

  Returns:
    function: right hand side function of the PDE given the field values
  """
  (X, Y), stepsize = get_grid(cfg)
  ddx = derivs.fourier_deriv(n_modes=cfg.axis_points, stepsize=stepsize, axis=0)
  ddy = derivs.fourier_deriv(n_modes=cfg.axis_points, stepsize=stepsize, axis=1)

  def rhs(field):
    return X * ddy(field) - Y * ddx(field)

  return rhs
