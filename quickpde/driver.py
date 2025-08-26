import logging

import h5py
import hydra
import jax

from quickpde.config import Config
from quickpde.pdes import PDE
from quickpde.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration
def main(cfg: Config) -> None:
  if cfg.use_double_precision:
    jax.config.update('jax_enable_x64', True)
  pde = PDE.from_config(cfg)
  trajectory, timepoints = pde.solve(cfg)
  jax.block_until_ready(trajectory)
  logging.info(
      'Generated trajectory with shape %s and dtype=%s',
      trajectory.shape,
      trajectory.dtype,
  )
  with h5py.File(cfg.outfile, 'w') as f:
    logging.info('Storing trajectory with shape %s', trajectory.shape)
    f.create_dataset('data', data=trajectory)
    f.create_dataset('time', data=timepoints)
  pass


if __name__ == "__main__":
  main()
