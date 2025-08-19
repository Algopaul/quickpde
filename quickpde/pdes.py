import jax.numpy as jnp

from quickpde import derivs
from quickpde.config import Config
from quickpde.grid import get_grid


def get_pde(cfg: Config):
  pde_dict = {
      'rotation_2d': rotation_2d,
      'rde_1d': rde_1d,
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


def rde_1d(cfg: Config):
  """Rotating detonation waves in 1D

  Args:
    cfg (Config): Config struct

  Returns:
    function: right hand side function of the PDE given the field values
  """
  _, stepsize = get_grid(cfg)
  # ddx = derivs.fourier_deriv(n_modes=cfg.axis_points, stepsize=stepsize, axis=0)
  ddx = derivs.upwind_deriv(stepsize, axis=0)
  d2dx2 = derivs.central_deriv(stepsize=stepsize, axis=0, order=2)

  nu = cfg.viscosity
  mu = cfg.injection_rate

  def omega(eta):
    eta_c = 1.1
    alpha = 0.3
    return jnp.exp((eta - eta_c) / alpha)

  def xi(x):
    epsilon = 0.11
    return -epsilon * x

  def beta(eta):
    eta_p = 0.5
    return mu / (1 + jnp.exp(5 * (eta - eta_p)))

  def rhs(state):
    eta, lam = jnp.split(state, 2)
    d_eta = -eta * ddx(eta) + nu * d2dx2(eta) + (1 - lam) * omega(eta) + xi(eta)
    d_lam = nu * d2dx2(lam) + (1 - lam) * omega(eta) - beta(eta) * lam
    return jnp.hstack((d_eta, d_lam))

  return rhs
