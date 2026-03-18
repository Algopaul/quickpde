# Deprecated: all functions have been merged into quickpde.diagnostics.
# This module exists only for backwards compatibility.
from quickpde.diagnostics import (
    gaussian_blur_fft_jax,
    top2_indices,
    enforce_min_separation_top2,
    find_top2_peaks_fft_jax,
    core_distance_jax,
    core_distance_jit,
    core_distance_batch,
)

__all__ = [
    'gaussian_blur_fft_jax',
    'top2_indices',
    'enforce_min_separation_top2',
    'find_top2_peaks_fft_jax',
    'core_distance_jax',
    'core_distance_jit',
    'core_distance_batch',
]
