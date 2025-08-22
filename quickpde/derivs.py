import jax.numpy as jnp


def upwind_deriv(stepsize, axis):
  """Upwind derivative (given direction)"""

  def deriv(field):
    fm = jnp.roll(field, 1, axis=axis)
    return (field - fm) / stepsize

  return deriv


def central_deriv(stepsize, axis, order=1):
  """Central difference with periodic wrap.

  Args:
    stepsize (float): grid spacing along 'axis'
    axis (int): axis to differentiate along
  """

  if order == 1:

    def deriv(field):
      fp = jnp.roll(field, -1, axis=axis)
      fm = jnp.roll(field, 1, axis=axis)
      return (fp - fm) / (2.0 * stepsize)
  elif order == 2:

    def deriv(field):
      fp = jnp.roll(field, -1, axis=axis)
      fm = jnp.roll(field, 1, axis=axis)
      return (-2 * field + fp + fm) / (stepsize**2)

  return deriv


def wave_numbers(n_modes, stepsize):
  return (2j * jnp.pi) * jnp.fft.fftfreq(n_modes, d=stepsize)


def fourier_deriv(n_modes, stepsize, axis, order=1):
  """Spectral derivative via FFT (periodic).
  
  Args:
    n_modes (int): number of grid points along 'axis'
    stepsize (float): grid spacing along 'axis'
    axis (int): axis to differentiate along
    order (int): derivative order (1 or 2)
  """
  if order not in (1, 2):
    raise NotImplementedError(
        "Fourier derivs only implemented for order 1 or 2")

  # Base multiplier for d/dx in Fourier space is (i * 2π * f)
  base_k = wave_numbers(n_modes, stepsize)
  if order == 1:
    kvec = base_k
  else:  # order == 2
    kvec = base_k**2  # (i 2π f)^2 = -(2π f)^2

  def ddx(field):
    if field.shape[axis] != n_modes:
      raise ValueError(
          f"Size mismatch along axis {axis}: field.shape[{axis}]={field.shape[axis]} != n_modes={n_modes}"
      )

    # Reshape kvec to broadcast along the chosen axis
    shape = [1] * field.ndim
    shape[axis] = n_modes
    K = jnp.reshape(kvec, shape)  # (..., n_modes, ...) with ones elsewhere

    fh = jnp.fft.fft(field, axis=axis)
    out = jnp.fft.ifft(K * fh, axis=axis)

    # For real inputs, the derivative is real (up to roundoff)
    return jnp.real(out)

  return ddx


def periodic_poisson_solver(n_modes, stepsize):
  K = wave_numbers(n_modes, stepsize)
  Ksq = K**2
  ones = jnp.ones((n_modes, n_modes))
  K2 = -Ksq[:, None] * ones - Ksq[None, :] * ones
  invK2 = 1 / K2
  invK2 = invK2.at[0, 0].set(0.0)

  def solver(field):
    return -jnp.real(jnp.fft.ifft2(invK2 * jnp.fft.fft2(field)))

  return solver
