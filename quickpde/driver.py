import logging
import os
from hashlib import md5
from pathlib import Path

import h5py
import hydra
import jax
from omegaconf import OmegaConf

from quickpde.config import Config
from quickpde.pdes import PDE
from quickpde.util import log_duration


def get_filename(cfg: Config):
  if cfg.outfile is None:
    s = md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:10]
  else:
    s = cfg.outfile
  s = s if s.endswith('.h5') else s + '.h5'
  full_path = Path(cfg.outdir) / s
  full_path.parent.mkdir(parents=True, exist_ok=True)
  return str(full_path)


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
  outfile = get_filename(cfg)
  with h5py.File(outfile, 'w') as f:
    logging.info(
        'Storing trajectory at %s with shape %s',
        outfile,
        trajectory.shape,
    )
    f.create_dataset('data', data=trajectory, dtype=cfg.store_type)
    f.create_dataset('time', data=timepoints, dtype=cfg.store_type)
  pass


if __name__ == "__main__":
  main()
