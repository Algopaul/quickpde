from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

import quickpde.initial_conditions as ic
from quickpde import derivs
from quickpde.config import Config
from quickpde.grid import get_grid
from quickpde.odesolve import get_ode_solver, get_rk


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

  def get_step(self, cfg: Config):
    return get_rk(self.rhs, cfg.dt)

  def solve(self, cfg: Config):
    x0 = self.initial_condition(cfg)
    if self.solver is None:
      step = self.get_step(cfg)
      self.solver = get_ode_solver(step, cfg)
    return self.solver(x0)


@PDE.register('allen_cahn')
class AllenCahn(PDE):
  """Allen-Cahn equation with time-dependent potential on [0, 1].

  ∂u/∂t = ε ∂²u/∂x² - a(t, x) * f(u)

  where f(u) = u - u³ and a(t, x) = 1.05 + t * sin(2π x).

  Time is augmented into the state as the last element so that the autonomous
  ODE solver can handle the explicit time dependence.
  """

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def get_rhs(self, cfg: Config):
    x, stepsize = get_grid(cfg)
    assert isinstance(x, jax.Array)
    d2dx2 = derivs.fourier_deriv(
        n_modes=cfg.axis_points, stepsize=stepsize, axis=0, order=2)
    eps = cfg.allen_cahn.epsilon

    def rhs(state):
      u, t = state[:-1], state[-1]
      a = 1.05 + t * jnp.sin(2 * jnp.pi * x)
      du = eps * d2dx2(u) - a * (u - u**3)
      return jnp.append(du, 1.0)

    return rhs

  def initial_condition(self, cfg: Config):
    x, _ = get_grid(cfg)
    assert isinstance(x, jax.Array)
    w = jnp.sqrt(20.0)
    phi_g = lambda b: jnp.exp(-w**2 * jnp.abs(jnp.sin(jnp.pi * (x - b)))**2)
    u0 = phi_g(0.03) - phi_g(0.7)
    return jnp.append(u0, 0.0)

  def solve(self, cfg: Config):
    trajectory, timepoints = super().solve(cfg)
    return trajectory[..., :-1], timepoints


@PDE.register('schroedinger')
class Schroedinger(PDE):
  """Schrödinger equation with quartic potential on [-10, 10].

  Splits the complex wavefunction into real and imaginary parts:
    state = [phi_r, phi_i]  (2N elements)

  d(phi_r)/dt = -0.5 * d²(phi_i)/dx² + V(x) * phi_i
  d(phi_i)/dt =  0.5 * d²(phi_r)/dx² - V(x) * phi_r

  where V(x) = alpha_2 * x² + alpha_4 * x⁴, alpha_2 = -1/8, alpha_4 = 1/64.
  """

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def get_rhs(self, cfg: Config):
    x, stepsize = get_grid(cfg)
    assert isinstance(x, jax.Array)
    d2dx2 = derivs.fourier_deriv(
        n_modes=cfg.axis_points, stepsize=stepsize, axis=0, order=2)
    alpha_2 = -1 / 8
    alpha_4 = alpha_2**2
    pot = alpha_2 * x**2 + alpha_4 * x**4

    def rhs(state):
      phi_r, phi_i = jnp.split(state, 2)
      phi_r_dot = -0.5 * d2dx2(phi_i) + pot * phi_i
      phi_i_dot = 0.5 * d2dx2(phi_r) - pot * phi_r
      return jnp.hstack((phi_r_dot, phi_i_dot))

    return rhs

  def initial_condition(self, cfg: Config):
    x, _ = get_grid(cfg)
    assert isinstance(x, jax.Array)
    q_l = -2.0
    phi_r0 = jnp.pi**(-0.25) * jnp.exp(-(x - q_l)**2 / 2)
    phi_i0 = jnp.zeros_like(phi_r0)
    return jnp.hstack((phi_r0, phi_i0))


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
    return ic.init_bump(cfg)


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
    mu = cfg.rde.injection_rate
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


@PDE.register('swe2d')
class ShallowWater(PDE):

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def initial_condition(self, cfg: Config):
    n = cfg.axis_points
    (X, Y), _ = get_grid(cfg)
    h = 1 + 0.33 * ic.gaussian_bump_2d(X, Y, [0.0, 0.0], cfg)
    phi = 0 * X
    return jnp.stack((h, phi))

  def get_rhs(self, cfg: Config):
    n = cfg.axis_points
    _, stepsize = get_grid(cfg)
    d_dx = derivs.fourier_deriv(n, stepsize, 0)
    d_dy = derivs.fourier_deriv(n, stepsize, 1)

    def rhs(state):
      h, phi = state
      dh_dt = -(d_dx(h * d_dx(phi)) + d_dy(h * d_dy(phi)))
      dphi_dt = -0.5 * (d_dx(phi)**2 + d_dy(phi)**2) - h
      return jnp.stack((dh_dt, dphi_dt))

    return rhs


@PDE.register('wave2d')
class Wave(PDE):

  def __init__(self, cfg: Config):
    self.rhs = self.get_rhs(cfg)

  def initial_condition(self, cfg: Config):
    n = cfg.axis_points
    (X, Y), _ = get_grid(cfg)
    rho0 = ic.gaussian_bump_2d(X, Y, [0.0, 0], cfg)
    vx = 0 * X
    vy = 0 * Y
    return jnp.stack((rho0, vx, vy))

  def get_rhs(self, cfg: Config):
    n = cfg.axis_points
    _, stepsize = get_grid(cfg)
    d_dx = derivs.central_deriv(stepsize, 0)
    d_dy = derivs.central_deriv(stepsize, 1)

    def rhs(state):
      rho, vx, vy = state
      drho_dt = -(d_dx(vx) + d_dy(vy))
      dvx_dt = -d_dx(rho)
      dvy_dt = -d_dy(rho)
      return jnp.stack((drho_dt, dvx_dt, dvy_dt))

    return rhs


@PDE.register('vorticity')
class Vorticity(PDE):

  def initial_condition(self, cfg: Config):
    if cfg.vorticity.initial == 'random':
      n = cfg.axis_points
      return ic.gaussian_random_field(
          1.0,
          n,
          n,
          cfg.vorticity.random_freq_decay,
          cfg.vorticity.random_seed,
      )
    elif cfg.vorticity.initial == 'twobump':
      return ic.double_bump(cfg)
    else:
      raise ValueError(f'Unknown vorticity initial condition: {cfg.vorticity.initial}')

  def get_step(self, cfg: Config):

    n = cfg.axis_points
    dx = (cfg.bound_x[1] - cfg.bound_x[0]) / n
    dt = cfg.dt
    (_, Y), _ = get_grid(cfg)

    ddx = derivs.fourier_deriv(n, dx, axis=0)
    ddy = derivs.fourier_deriv(n, dx, axis=1)
    poisson_solve = derivs.periodic_poisson_solver(n, dx)
    K = derivs.wave_numbers(n, dx)
    kx = K[:, None]
    ky = K[None, :]
    ksq = kx**2 + ky**2
    ksq_hyp = jnp.abs(ksq)
    ksq_hyp /= jnp.max(ksq_hyp)

    hyperviscosity_mag = cfg.vorticity.hyperviscosity_mag
    hyperviscosity_exp = cfg.vorticity.hyperviscosity_exp

    def viscosity_term(fhat):
      return cfg.viscosity * jnp.real(jnp.fft.ifft2(ksq * fhat))

    def hyperviscosity_term(fhat):
      damping = -ksq_hyp**hyperviscosity_exp
      return hyperviscosity_mag * jnp.real(jnp.fft.ifft2(damping * fhat))

    def rhsuv(field, u, v):
      if cfg.vorticity.forcing:
        force = 1.0 * jnp.cos(4 * Y)
      else:
        force = 0.0
      return -(u * ddx(field) + v * ddy(field)) + force

    @jax.jit
    def rk4_mod(i, field):
      del i
      psi = poisson_solve(field)
      u = ddy(psi)
      v = -ddx(psi)
      k1 = rhsuv(field, u, v)
      k2 = rhsuv(field + 0.5 * dt * k1, u, v)
      k3 = rhsuv(field + 0.5 * dt * k2, u, v)
      k4 = rhsuv(field + dt * k3, u, v)
      f = field + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
      fhat = jnp.fft.fft2(field)
      f = f + dt * viscosity_term(fhat)
      fhat = jnp.fft.fft2(f)
      return f + dt * hyperviscosity_term(fhat)

    return rk4_mod
