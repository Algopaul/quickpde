import jax.numpy as jnp
from hydra.core.config_store import ConfigStore

from quickpde.config.base import Config

default_rotation = Config(
    'rotation',
    pde='rotation_2d',
    axis_points=128,
    bound_x=(-jnp.pi, jnp.pi),
    bound_y=(-jnp.pi, jnp.pi),
    dt=1e-3 * jnp.pi,
    t_end=jnp.pi,
    ic_sharpness=80,
)

default_rde = Config(
    'rde',
    pde='rde_1d',
    domain_dim=1,
    axis_points=512,
    bound_x=(0, 2 * jnp.pi),
    dt=1e-3,
    t_end=50,
    viscosity=1e-4,
)

cs = ConfigStore.instance()
cs.store(name="rotation", node=default_rotation)
cs.store(name="rde", node=default_rde)
