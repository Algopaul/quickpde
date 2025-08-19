import jax.numpy as jnp
from hydra.core.config_store import ConfigStore

from quickpde.config.base import Config

default_rotation = Config(
    'rotation',
    axis_points=128,
    bound_x=(-jnp.pi, jnp.pi),
    bound_y=(-jnp.pi, jnp.pi),
    dt=1e-3 * jnp.pi,
    t_end=jnp.pi,
)

cs = ConfigStore.instance()
cs.store(name="rotation", node=default_rotation)
