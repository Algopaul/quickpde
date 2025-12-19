import jax.numpy as jnp


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
