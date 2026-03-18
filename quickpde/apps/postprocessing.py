from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import zarr
from tqdm import tqdm

from quickpde import diagnostics as diag
from quickpde.config import Config
from quickpde.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../../conf')
@log_duration
def main(cfg: Config) -> None:
  files = list(Path(cfg.outdir).rglob('*.zarr'))
  stepsize = (cfg.bound_x[1] - cfg.bound_x[0]) / cfg.axis_points
  fields = {'energy': [], 'enstrophy': [], 'rfe': []}
  rfe = diag.RadialFrequencyEnergies(cfg.axis_points, 8)
  rfe_fun = lambda x, _: jax.jit(rfe.radial_spectrum)(x)
  for file in tqdm(files):
    root = zarr.open_group(str(file), mode='a')
    data = jnp.array(root['data'])
    for field, fn in zip(['energy', 'enstrophy', 'rfe'],
                         [diag.energy, diag.enstrophy, rfe_fun]):
      val = jax.vmap(fn, in_axes=(0, None))(data, stepsize)
      root.create_array(field, data=np.array(val), overwrite=True)
      fields[field].append(val)

  stacked = {k: jnp.stack(v) for k, v in fields.items()}

  outdir = Path(cfg.outdir)
  for k, v in stacked.items():
    mu = np.array(jnp.mean(v, axis=0))
    var = np.array(jnp.var(v, axis=0))
    summary_path = outdir.parent / f'{outdir.name}_{k}.zarr'
    root = zarr.open_group(str(summary_path), mode='w')
    root.create_array('mean', data=mu)
    root.create_array('var', data=var)
    root.create_array('vals', data=np.array(v))
    if v.ndim == 2:
      hist_vals, bins = jnp.histogram(v[:, 0], 32)
      root.create_array('hist/vals', data=np.array(hist_vals))
      root.create_array('hist/bins', data=np.array(bins))


if __name__ == "__main__":
  main()
