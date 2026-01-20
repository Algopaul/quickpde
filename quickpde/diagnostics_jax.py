import jax
import jax.numpy as jnp


def gaussian_blur_fft_jax(f, sigma, dx=1.0):
  """
    Periodic Gaussian blur of a 2D field using FFT (JAX).
    """
  nx, ny = f.shape
  kx = jnp.fft.fftfreq(nx, d=dx)
  ky = jnp.fft.fftfreq(ny, d=dx)
  KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

  G = jnp.exp(-2.0 * (jnp.pi**2) * sigma**2 * (KX**2 + KY**2))
  return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(f) * G))


def top2_indices(f):
  """
    Return top-2 values and indices of a 2D array.
    Shapes are always static: (2,)
    """
  flat = f.reshape(-1)
  idx = jnp.argpartition(flat, -2)[-2:]  # unsorted
  vals = flat[idx]

  order = jnp.argsort(vals)[::-1]  # descending
  idx = idx[order]
  vals = vals[order]

  ny = f.shape[1]
  i = idx // ny
  j = idx % ny
  return vals, i, j


def enforce_min_separation_top2(vals, i, j, r_min):
  """
    If the second peak is closer than r_min, collapse it onto the first.
    Keeps shapes static and is JIT-safe.
    """
  di = i[1] - i[0]
  dj = j[1] - j[0]
  dist2 = di * di + dj * dj

  keep = dist2 >= (r_min * r_min)

  vals = jnp.where(keep, vals, jnp.array([vals[0], 0.0]))
  i = jnp.where(keep, i, jnp.array([i[0], i[0]]))
  j = jnp.where(keep, j, jnp.array([j[0], j[0]]))
  return vals, i, j


def find_top2_peaks_fft_jax(f, sigma=2.0, r_min=2, dx=1.0):
  f_s = gaussian_blur_fft_jax(f, sigma=sigma, dx=dx)
  vals, i, j = top2_indices(f_s)
  vals, i, j = enforce_min_separation_top2(vals, i, j, r_min)
  return vals, i, j


def core_distance_jax(field, dx=1.0, sigma=2.0, r_min=2):
  """
    Distance between the two strongest (separated) peaks.
    Returns 0 if only one valid peak exists.
    """
  _, i, j = find_top2_peaks_fft_jax(field, sigma=sigma, r_min=r_min, dx=dx)

  dxp = dx * (i[1] - i[0])
  dyp = dx * (j[1] - j[0])
  dist = jnp.sqrt(dxp * dxp + dyp * dyp)

  valid = (i[1] != i[0]) | (j[1] != j[0])
  return jnp.where(valid, dist, 0.0)


# JIT + VMAP versions
core_distance_jit = jax.jit(core_distance_jax)

core_distance_batch = jax.jit(
    jax.vmap(core_distance_jax, in_axes=(0, None, None, None)))

# Example shapes:
# field        : (H, W)
# field_batch  : (B, H, W)
