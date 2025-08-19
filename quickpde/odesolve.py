import jax
import jax.numpy as jnp

from quickpde.config import Config


def get_ode_solver(rhs, cfg: Config):
  step = get_rk(rhs, cfg.dt)
  timepoints = jnp.arange(0, cfg.t_end, cfg.dt)

  def solve(x0):

    def inner(field, t):
      del t
      out = jax.lax.fori_loop(0, cfg.store_every, step, field)
      return out, out

    _, traj = jax.lax.scan(inner, x0, timepoints[::cfg.store_every])
    return traj, timepoints[::cfg.store_every] + cfg.dt * cfg.store_every

  return solve


def get_rk(rhs, dt):

  @jax.jit
  def rk4_mod(i, field):
    del i
    k1 = rhs(field)
    k2 = rhs(field + 0.5 * dt * k1)
    k3 = rhs(field + 0.5 * dt * k2)
    k4 = rhs(field + dt * k3)
    f = field + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return f

  return rk4_mod
