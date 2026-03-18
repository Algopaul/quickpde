# quickpde

A lightweight package for generating PDE solution trajectories. Built for use as training data in scientific machine learning.

- Minimal abstractions: PDE structure is not hidden behind layers of framework code
- [JAX](https://github.com/google/jax)-based numerics with JIT compilation and GPU support
- Hydra configuration with CLI overrides and multi-run sweeps
- Zarr output format for efficient storage and access

## Supported PDEs

| Preset | PDE | Domain | Fields |
|---|---|---|---|
| `rotation` | Rotation 2D | 2D, periodic | 1 |
| `rde` | Rotating Detonation Waves | 1D, periodic | 2 |
| `wave2d` | Linear Wave Equation | 2D, periodic | 3 |
| `swe2d` | Shallow Water Equations | 2D, periodic | 2 |
| `vorticity` | Vorticity Transport (two-bump IC) | 2D, periodic | 1 |
| `vorticity_grf` | Vorticity Transport (random field IC) | 2D, periodic | 1 |

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

## Running a simulation

```bash
python quickpde/driver.py -cn <preset> [OVERRIDES]
```

Output is written as a Zarr group to `data/results/` by default. The group contains:
- `data`: trajectory array (shape depends on PDE)
- `time`: time points corresponding to stored snapshots

**Example** — run the two-bump vorticity preset at lower resolution:

```bash
python quickpde/driver.py -cn vorticity axis_points=128 t_end=50
```

**Example** — generate an ensemble with different random seeds:

```bash
python quickpde/driver.py --multirun -cn vorticity_grf vorticity.random_seed="range(0,10)"
```

## Configuration

All parameters can be overridden on the command line. The table below lists every option.

### Common parameters

| Parameter | Default | Description |
|---|---|---|
| `axis_points` | preset-specific | Number of grid points per axis |
| `dt` | preset-specific | Time step size |
| `t_end` | preset-specific | End time of simulation |
| `store_every` | `1` | Save a snapshot every N time steps |
| `ic_sharpness` | preset-specific | Controls width of initial bump (larger = sharper) |
| `viscosity` | preset-specific | Kinematic viscosity ν |
| `outdir` | `data/results` | Output directory |
| `outfile` | auto-generated | Output filename (Zarr group) |
| `store_type` | `f4` | Floating-point precision for stored data (`f4` or `f8`) |
| `use_double_precision` | `False` | Enable 64-bit arithmetic in JAX |

### RDE-specific parameters (`rde.*`)

| Parameter | Default | Description |
|---|---|---|
| `rde.injection_rate` | `3.0` | Injection rate μ; controls number of detonation waves |

### Vorticity-specific parameters (`vorticity.*`)

| Parameter | Default | Description |
|---|---|---|
| `vorticity.initial` | preset-specific | Initial condition type: `random` or `twobump` |
| `vorticity.random_freq_decay` | `2.0` | Spectral decay exponent α for random field (k⁻ᵅ) |
| `vorticity.random_seed` | `0` | Random seed for reproducible random-field ICs |
| `vorticity.bump_distance` | `1.0` | Distance between the two vortex bumps |
| `vorticity.bump_angle` | `0.0` | Rotation angle of the two-bump configuration |
| `vorticity.forcing` | `False` | Add cosine forcing term |
| `vorticity.hyperviscosity_mag` | `500` | Hyperviscosity coefficient κ |
| `vorticity.hyperviscosity_exp` | `8` | Hyperviscosity order (∇^n applied to ω) |

## Post-processing

The `analysis` CLI provides diagnostics on saved Zarr files:

```bash
# Compute enstrophy over time
analysis enstrophies file="data/results/output.zarr" datakey="data" stepsize=0.098

# Compute energy spectra
analysis spectrum file="data/results/output.zarr" datakey="data" nbins=16 n_samples=4
```

## Numerical methods

- **Time integration**: explicit RK4
- **Spatial derivatives**: Fourier spectral (all PDEs use periodic domains)
- **Poisson solver**: spectral, used for vorticity stream-function inversion

## Documentation

Full per-PDE documentation with equations and citations lives in `docs/`. Build with:

```bash
sphinx-autobuild docs docs/_build/html
```
