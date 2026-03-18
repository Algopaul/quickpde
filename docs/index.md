# quickpde Documentation

```{toctree}
:maxdepth: 2
:hidden:

api/index
api/modules
RDE
Rotation2d
Vorticity
ShallowWater2d
Wave2d
```

A lightweight package for generating PDE solution trajectories, built for use as training data in scientific machine learning.

- Minimal abstractions — PDE structure is not hidden behind layers of framework code
- [JAX](https://github.com/google/jax)-based numerics with JIT compilation and GPU support
- Hydra configuration with CLI overrides and multi-run sweeps
- Zarr output for efficient storage and access

## Quick start

```bash
pip install -e ".[dev]"
python quickpde/driver.py -cn rotation
```

## Supported PDEs

| Page | Preset | Description |
|---|---|---|
| [Rotation 2D](Rotation2d) | `rotation` | Linear transport, rotating Gaussian bump |
| [Rotating Detonation Waves](RDE) | `rde` | Nonlinear 1D shock waves, combustion model |
| [Wave Equation 2D](Wave2d) | `wave2d` | Linear acoustic waves, Hamiltonian |
| [Shallow Water 2D](ShallowWater2d) | `swe2d` | Nonlinear dispersive waves, Hamiltonian |
| [Vorticity Transport](Vorticity) | `vorticity`, `vorticity_grf` | 2D incompressible flow |

Every PDE has default initial conditions adjustable via `ic_sharpness`.
The vorticity PDE additionally supports random Gaussian-field initialization for dataset generation.
