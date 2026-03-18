import jax
import jax.numpy as jnp
import numpy as np


def wave_numbers(n_modes, stepsize):
  return (2 * jnp.pi) * jnp.fft.fftfreq(n_modes, d=stepsize)


def wave_numbers_2d(n, stepsize):
  k1d = wave_numbers(n, stepsize=stepsize)
  kx, ky = jnp.meshgrid(k1d, k1d, indexing='ij')
  k2 = kx**2 + ky**2
  return kx, ky, k2


def velocity(omega, stepsize):
  n = omega.shape[0]
  kx, ky, k2 = wave_numbers_2d(n, stepsize)
  omega_hat = jnp.fft.fft2(omega)
  psi_hat = jnp.where(k2 == 0, 0.0, -omega_hat / k2)
  u_hat = 1j * ky * psi_hat
  v_hat = -1j * kx * psi_hat
  u = jnp.real(jnp.fft.ifft2(u_hat))
  v = jnp.real(jnp.fft.ifft2(v_hat))
  return u, v


def energy(omega, stepsize):
  u, v = velocity(omega, stepsize)
  return 0.5 * jnp.sum(u**2 + v**2) * stepsize**2


def enstrophy(omega, stepsize):
  return 0.5 * jnp.sum(omega**2) * stepsize**2


def divergence_error(omega, stepsize):
  N = omega.shape[0]
  kx, ky, _ = wave_numbers_2d(N, stepsize)

  omega_hat = jnp.fft.fft2(omega)
  psi_hat = jnp.where(kx**2 + ky**2 == 0, 0.0, -omega_hat / (kx**2 + ky**2))

  u_hat = 1j * ky * psi_hat
  v_hat = -1j * kx * psi_hat

  div_hat = 1j * kx * u_hat + 1j * ky * v_hat
  div = jnp.real(jnp.fft.ifft2(div_hat))

  return {
      "L2": jnp.sqrt(jnp.mean(div**2)),
      "Linf": jnp.max(jnp.abs(div)),
  }


def energy_spectrum(omega, stepsize, nbins=None):
  N = omega.shape[0]
  if nbins is None:
    nbins = N // 2

  kx, ky, k2 = wave_numbers_2d(N, stepsize)
  k = jnp.sqrt(k2)
  omega_hat = jnp.fft.fft2(omega)
  psi_hat = jnp.where(k2 == 0, 0.0, -omega_hat / k2)
  u_hat = 1j * ky * psi_hat
  v_hat = -1j * kx * psi_hat
  E_k = 0.5 * (jnp.abs(u_hat)**2 + jnp.abs(v_hat)**2) / N**2
  k_flat = k.flatten()
  E_flat = E_k.flatten()
  kmax = jnp.max(k_flat)
  bins = jnp.linspace(0.0, kmax, nbins + 1)
  spec = []
  k_mid = []
  for i in range(nbins):
    mask = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
    spec.append(jnp.sum(E_flat[mask]))
    k_mid.append(0.5 * (bins[i] + bins[i + 1]))

  return jnp.array(k_mid), jnp.array(spec)


class RadialFrequencyEnergies:

  def __init__(self, datadim, nbins, kmin=1.0):
    self.datadim = datadim
    self.nbins = nbins
    shell_idx, _ = self._radial_shells_log_bins(datadim, nbins, kmin)
    self.radial_spectrum = lambda x: self._radial_energy_spectrum(
        x, shell_idx, nbins)

  @staticmethod
  def _radial_shells_log_bins(N: int, nbins: int, kmin=1.0):
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

  def _radial_energy_spectrum(self, u, shell_idx, nbins):
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


def gaussian_blur_fft(f, sigma, dx=1.0):
  """
    Periodic Gaussian blur of a 2D field using FFT.
    sigma is in physical units (same as dx).
    """
  nx, ny = f.shape

  kx = np.fft.fftfreq(nx, d=dx)
  ky = np.fft.fftfreq(ny, d=dx)
  KX, KY = np.meshgrid(kx, ky, indexing="ij")

  G = np.exp(-2 * (np.pi**2) * sigma**2 * (KX**2 + KY**2))
  return np.real(np.fft.ifft2(np.fft.fft2(f) * G))


def topk_numpy(f, k):
  flat = f.ravel()
  idx = np.argpartition(flat, -k)[-k:]  # unsorted
  vals = flat[idx]

  ny = f.shape[1]
  i = idx // ny
  j = idx % ny

  # sort descending
  order = np.argsort(vals)[::-1]
  return vals[order], i[order], j[order]


def enforce_min_separation(i, j, vals, r_min):
  keep_i = []
  keep_j = []
  keep_v = []

  for ii, jj, v in zip(i, j, vals):
    ok = True
    for ki, kj in zip(keep_i, keep_j):
      if (ii - ki)**2 + (jj - kj)**2 < r_min**2:
        ok = False
        break
    if ok:
      keep_i.append(ii)
      keep_j.append(jj)
      keep_v.append(v)

  return (
      np.array(keep_v),
      np.array(keep_i),
      np.array(keep_j),
  )


def find_topk_peaks_fft(
    f,
    k=2,
    sigma=2.0,
    r_min=2,
    dx=1.0,
):
  f_s = gaussian_blur_fft(f, sigma=sigma, dx=dx)
  vals, i, j = topk_numpy(f_s, k)
  vals, i, j = enforce_min_separation(i, j, vals, r_min)
  return vals, i, j


def core_distance(field, dx=1.0):
  x, y = find_topk_peaks_fft(field)[1:]
  if len(x) > 1:
    delta_x = dx * (x[1] - x[0])
    delta_y = dx * (y[1] - y[0])
    return np.sqrt(delta_x**2 + delta_y**2)
  else:
    return 0.0


# ---------------------------------------------------------------------------
# JAX-compatible (JIT/vmap-safe) peak-finding
# ---------------------------------------------------------------------------

def gaussian_blur_fft_jax(f, sigma, dx=1.0):
  """Periodic Gaussian blur of a 2D field using FFT (JAX/JIT-compatible)."""
  nx, ny = f.shape
  kx = jnp.fft.fftfreq(nx, d=dx)
  ky = jnp.fft.fftfreq(ny, d=dx)
  KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
  G = jnp.exp(-2.0 * (jnp.pi**2) * sigma**2 * (KX**2 + KY**2))
  return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(f) * G))


def top2_indices(f):
  """Return top-2 values and (i, j) indices of a 2D array. Shapes are static: (2,)."""
  flat = f.reshape(-1)
  idx = jnp.argpartition(flat, -2)[-2:]
  vals = flat[idx]
  order = jnp.argsort(vals)[::-1]
  idx = idx[order]
  vals = vals[order]
  ny = f.shape[1]
  i = idx // ny
  j = idx % ny
  return vals, i, j


def enforce_min_separation_top2(vals, i, j, r_min):
  """Collapse the second peak onto the first if they are closer than r_min."""
  di = i[1] - i[0]
  dj = j[1] - j[0]
  keep = di * di + dj * dj >= r_min * r_min
  vals = jnp.where(keep, vals, jnp.array([vals[0], 0.0]))
  i = jnp.where(keep, i, jnp.array([i[0], i[0]]))
  j = jnp.where(keep, j, jnp.array([j[0], j[0]]))
  return vals, i, j


def find_top2_peaks_fft_jax(f, sigma=2.0, r_min=2, dx=1.0):
  f_s = gaussian_blur_fft_jax(f, sigma=sigma, dx=dx)
  vals, i, j = top2_indices(f_s)
  return enforce_min_separation_top2(vals, i, j, r_min)


def core_distance_jax(field, dx=1.0, sigma=2.0, r_min=2):
  """Distance between the two strongest separated peaks. Returns 0 if only one peak."""
  _, i, j = find_top2_peaks_fft_jax(field, sigma=sigma, r_min=r_min, dx=dx)
  dxp = dx * (i[1] - i[0])
  dyp = dx * (j[1] - j[0])
  dist = jnp.sqrt(dxp * dxp + dyp * dyp)
  valid = (i[1] != i[0]) | (j[1] != j[0])
  return jnp.where(valid, dist, 0.0)


core_distance_jit = jax.jit(core_distance_jax)

core_distance_batch = jax.jit(
    jax.vmap(core_distance_jax, in_axes=(0, None, None, None)))
