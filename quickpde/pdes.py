from abc import ABC, abstractmethod, abstractproperty
from typing import Callable

import jax
import jax.numpy as jnp

from quickpde import derivs
from quickpde.config import Config
from quickpde.grid import get_grid
from quickpde.odesolve import get_ode_solver


class PDE(ABC):
  rhs: Callable
  solver: Callable | None = None
  registry = {}

  @classmethod
  def from_config(cls, cfg: Config):
    kind = cfg.pde
    if kind not in cls.registry:
      raise ValueError(f'Unkown kind {kind}')
    return cls.registry[kind](cfg)

  @classmethod
  def register(cls, kind: str):

    def decorator(subclass):
      cls.registry[kind] = subclass
      return subclass

    return decorator

  @abstractmethod
  def initial_condition(self, cfg: Config) -> jax.Array:
    pass

  def solve(self, cfg: Config):
    x0 = self.initial_condition(cfg)
    if self.solver is None:
      self.solver = get_ode_solver(self.rhs, cfg)
    return self.solver(x0)


@PDE.register('rotation_2d')
class Rotation2d(PDE):

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def get_rhs(self, cfg: Config):
    (X, Y), stepsize = get_grid(cfg)
    ddx = derivs.fourier_deriv(
        n_modes=cfg.axis_points, stepsize=stepsize, axis=0)
    ddy = derivs.fourier_deriv(
        n_modes=cfg.axis_points, stepsize=stepsize, axis=1)

    def rhs(field):
      return X * ddy(field) - Y * ddx(field)

    return rhs

  def initial_condition(self, cfg):
    (X, Y), _ = get_grid(cfg)
    return jnp.exp(-cfg.ic_sharpness * ((X - 1.0)**2 + Y**2))


@PDE.register('rde_1d')
class RDE1d(PDE):

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def get_rhs(self, cfg: Config):
    _, stepsize = get_grid(cfg)
    ddx = derivs.upwind_deriv(stepsize, axis=0)
    d2dx2 = derivs.central_deriv(stepsize=stepsize, axis=0, order=2)

    # Parameters
    nu = cfg.viscosity
    mu = cfg.injection_rate
    # Constants (for now)
    eta_c = 1.1
    alpha = 0.3
    epsilon = 0.11
    eta_p = 0.5

    omega = lambda eta: jnp.exp((eta - eta_c) / alpha)
    xi = lambda eta: -epsilon * eta
    beta = lambda eta: mu / (1 + jnp.exp(5 * (eta - eta_p)))

    def rhs(state):
      eta, lam = jnp.split(state, 2)
      lam_term = (1 - lam) * omega(eta)
      d_eta = -eta * ddx(eta) + nu * d2dx2(eta) + lam_term + xi(eta)
      d_lam = nu * d2dx2(lam) + lam_term - beta(eta) * lam
      return jnp.hstack((d_eta, d_lam))

    return rhs

  def initial_condition(self, cfg: Config):
    x, _ = get_grid(cfg)
    assert isinstance(x, jax.Array)
    # eta_0 = jnp.exp(-cfg.ic_sharpness * (x - 1.0)**2)
    eta_0 = 3 / 2 * (1.0 / jnp.cosh(x - 1.0))**2
    lam_0 = jnp.zeros_like(x)
    return jnp.hstack((eta_0, lam_0))
