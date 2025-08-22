import logging

import h5py
import hydra
import jax
import jax.numpy as jnp

from quickpde.config import Config
from quickpde.pdes import PDE
from quickpde.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration
def main(cfg: Config) -> None:
  pde = PDE.from_config(cfg)
  trajectory, timepoints = pde.solve(cfg)
  jax.block_until_ready(trajectory)
  logging.info('Generated trajectory with shape %s', trajectory.shape)
  with h5py.File(f'data/test_{cfg.injection_rate:.2f}.h5', 'w') as f:
    # trajectory = jnp.reshape(
    #     trajectory, [trajectory.shape[0], 2, trajectory.shape[-1] // 2])
    # trajectory = jnp.transpose(trajectory, [0, 2, 1])
    # trajectory = jnp.stack(jnp.split(trajectory, 500))
    logging.info('Storing trajectory with shape %s', trajectory.shape)
    f.create_dataset('data', data=trajectory)
    f.create_dataset('time', data=timepoints)
  pass


if __name__ == "__main__":
  main()
