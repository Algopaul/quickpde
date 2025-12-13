import jax.numpy as jnp
from hydra.core.config_store import ConfigStore

from quickpde.config.base import Config, VorticityConfig

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

default_wave = Config(
    'wave',
    pde='wave2d',
    domain_dim=2,
    axis_points=512,
    bound_x=(-jnp.pi, jnp.pi),
    dt=1e-3,
    t_end=8,
    viscosity=0.0,
)

swe = Config(
    'swe',
    pde='swe2d',
    domain_dim=2,
    axis_points=256,
    bound_x=(-jnp.pi, jnp.pi),
    bound_y=(-jnp.pi, jnp.pi),
    dt=1e-3,
    t_end=10,
    viscosity=0.0,
    store_every=80,
    ic_sharpness=2.7,
)

default_vorticity = Config(
    'vorticity',
    pde='vorticity',
    domain_dim=2,
    viscosity=0.0,
    axis_points=256,
    bound_x=(-2 * jnp.pi, 2 * jnp.pi),
    bound_y=(-2 * jnp.pi, 2 * jnp.pi),
    dt=2e-3,
    t_end=200,
    store_every=200,
    ic_sharpness=5.0,
    vorticity=VorticityConfig('twobump', bump_distance=2.0),
)

vorticity_grf = Config(
    'vorticity',
    pde='vorticity',
    domain_dim=2,
    viscosity=0.0,
    axis_points=256,
    bound_x=(-2 * jnp.pi, 2 * jnp.pi),
    bound_y=(-2 * jnp.pi, 2 * jnp.pi),
    dt=2e-3,
    t_end=200,
    store_every=200,
    ic_sharpness=5.0,
    vorticity=VorticityConfig('twobump', bump_distance=2.0),
)

cs = ConfigStore.instance()
cs.store(name="rotation", node=default_rotation)
cs.store(name="rde", node=default_rde)
cs.store(name="vorticity", node=default_vorticity)
cs.store(name="wave2d", node=default_wave)
cs.store(name="swe2d", node=swe)
