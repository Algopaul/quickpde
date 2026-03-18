from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import typer
import zarr
from rich.console import Console

import quickpde.diagnostics as dgn

app = typer.Typer(help="Unix-style tools for working with Zarr stores")
console = Console()
err_console = Console(stderr=True)


@app.command()
def enstrophies(file, datakey, stepsize: float = 0.098, sample_axis: int = 0):
  # (n_trajectories, time, N, N, C)
  root = zarr.open_group(file, mode='a')
  data = jnp.array(root[datakey])

  if data.shape[-1] == 1:
    data = data[..., 0]
  fn = lambda x: dgn.enstrophy(x, stepsize)
  fn = jax.vmap(fn)
  if data.ndim == 4:  # ensemble
    fn = jax.vmap(fn)
  ens = fn(data)
  root.create_array('enstrophies', data=np.array(ens), overwrite=True)
  root.create_array('ens_mean', data=np.array(jnp.mean(ens, axis=sample_axis)), overwrite=True)
  root.create_array('ens_std', data=np.array(jnp.std(ens, axis=sample_axis)), overwrite=True)


def radial_shells_log_bins(N: int, nbins: int, kmin=1.0):
  k = jnp.fft.fftfreq(N) * N
  kx, ky = jnp.meshgrid(k, k, indexing="ij")
  r = jnp.sqrt(kx**2 + ky**2)
  r_max = jnp.max(r)
  edges = jnp.logspace(
      jnp.log10(kmin),
      jnp.log10(r_max),
      nbins + 1,
  )
  shell_idx = jnp.digitize(r, edges) - 1
  shell_idx = jnp.clip(shell_idx, 0, nbins - 1)
  return shell_idx.astype(jnp.int32), nbins


@partial(jax.jit, static_argnums=2)
def radial_energy_spectrum(u, shell_idx, nbins):
  U = jnp.fft.fftn(u)
  E = jnp.abs(U)**2 / u.size
  spec = jnp.bincount(
      shell_idx.ravel(),
      weights=E.ravel(),
      length=nbins,
  )
  counts = jnp.bincount(shell_idx.ravel(), length=nbins)
  spec = spec / jnp.maximum(counts, 1)
  return jnp.log(spec + 1e-12)


def frequency_energies(data, shell_idx, nbins):
  all_energies = []
  for d in data:
    energies = jax.vmap(
        radial_energy_spectrum, in_axes=(0, None, None))(
            d,
            shell_idx,
            nbins,
        )
    all_energies.append(energies)
  return jnp.stack(all_energies)


@app.command()
def spectrum(
    file,
    datakey: str = 'data',
    sample_axis: int = 0,
    nbins: int = 8,
    n_samples: int = 4,
):
  root = zarr.open_group(file, mode='a')
  data = jnp.array(root[datakey])
  idcs = jnp.floor(jnp.linspace(0, len(data) - 1, n_samples)).astype(jnp.int32)
  if sample_axis == 0:
    data = data[:, idcs, ...]
  else:
    data = data[idcs, ...]
  if data.shape[-1] == 1:
    data = data[..., 0]
  N = data.shape[-1]
  shell_idx = radial_shells_log_bins(N, nbins, 1.0)[0]
  energies = frequency_energies(data, shell_idx, nbins)
  root.create_array('eng_mean', data=np.array(jnp.mean(energies, axis=sample_axis)), overwrite=True)
  root.create_array('eng_std', data=np.array(jnp.std(energies, axis=sample_axis)), overwrite=True)


def main():
  app()
