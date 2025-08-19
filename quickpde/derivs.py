import jax.numpy as jnp


def central_deriv(stepsize, axis):
  """Central difference with periodic wrap.

  Args:
    stepsize (float): grid spacing along 'axis'
    axis (int): axis to differentiate along
  """

  def deriv(field):
    fp = jnp.roll(field, -1, axis=axis)
    fm = jnp.roll(field, 1, axis=axis)
    return (fp - fm) / (2.0 * stepsize)

  return deriv


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
  base_k = (2j * jnp.pi) * jnp.fft.fftfreq(n_modes, d=stepsize)
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
