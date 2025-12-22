import logging
import os
from hashlib import md5
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm

from quickpde import diagnostics as diag
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


@hydra.main(version_base=None, config_name='config', config_path='../../conf')
@log_duration
def main(cfg: Config) -> None:
  files = Path(cfg.outdir).rglob('*.h5')
  files = list(files)
  stepsize = 4 * jnp.pi / 128
  fields = {'energy': [], 'enstrophy': [], 'rfe': []}
  rfe = diag.RadialFrequencyEnergies(128, 8)
  rfe_fun = lambda x, _: jax.jit(rfe.radial_spectrum)(x)
  for file in tqdm(files):
    with h5py.File(file, 'a') as f:
      for field, fn in zip(['energy', 'enstrophy', 'rfe'],
                           [diag.energy, diag.enstrophy, rfe_fun]):
        if field in f:
          del f[field]
        val = jax.vmap(fn, in_axes=(0, None))(jnp.array(f['data']), stepsize)
        f[field] = val
        fields[field].append(val)

  stacked = {}
  for k, v in fields.items():
    stacked[k] = jnp.stack(v)

  # Summary
  for k, v in stacked.items():
    mu = jnp.mean(v, axis=0)
    var = jnp.var(v, axis=0)
    if v.ndim == 2:
      vals, bins = jnp.histogram(v[:, 0], 32)
    with h5py.File(cfg.outdir + f'_{k}.h5', 'w') as f:
      f.create_dataset('mean', data=mu)
      f.create_dataset('var', data=var)
      f.create_dataset('vals', data=v)
      if v.ndim == 2:
        f.create_dataset('hist/vals', data=vals)
        f.create_dataset('hist/bins', data=bins)


if __name__ == "__main__":
  main()
