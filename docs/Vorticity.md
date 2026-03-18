# Vorticity Transport 2d

- Domain: 2d, $[-2\pi, 2\pi)^2$
- Fields: 1 (vorticity $\omega$)
- Properties: Transport, nonlinear
- Modifiers: `viscosity`, `vorticity.initial`, `vorticity.hyperviscosity_mag`, `vorticity.forcing`

## Explanation

Spectral solver for the 2D vorticity transport equation

```{math}
:label: eq-vorticity

\partial_t \omega = (\boldsymbol{u} \cdot \nabla)\omega + \nu\nabla^2\omega - \kappa\nabla^8\omega,
```

in the domain $[-2\pi, 2\pi)^2$ with periodic boundary conditions. The velocity field $\boldsymbol{u} = [u_1, u_2]^\top$ is recovered from the vorticity via the stream function $\Psi$ satisfying $-\Delta\Psi = \omega$, with $u_1 = \partial_y\Psi$ and $u_2 = -\partial_x\Psi$.

The hyperviscosity term $-\kappa\nabla^8\omega$ damps spurious high-frequency noise without affecting the large-scale dynamics, which is useful for near-inviscid runs ($\nu \approx 0$).

### Initial conditions

Two options are available via `vorticity.initial`:

**`twobump`** (default for `-cn vorticity`): Two Gaussian vortex bumps placed symmetrically,

```{math}
\omega_0(\boldsymbol{x}) = \exp\!\bigl(-\alpha\|\boldsymbol{x} - \boldsymbol{c}_1\|^2\bigr) - \exp\!\bigl(-\alpha\|\boldsymbol{x} - \boldsymbol{c}_2\|^2\bigr),
```

where $\boldsymbol{c}_1, \boldsymbol{c}_2$ are controlled by `vorticity.bump_distance` and `vorticity.bump_angle`, and $\alpha$ is set by `ic_sharpness`.

**`random`** (default for `-cn vorticity_grf`): Gaussian random field with power-law spectrum,

```{math}
\hat\omega_0(\boldsymbol{k}) \sim \mathcal{N}(0,1) \cdot |\boldsymbol{k}|^{-\alpha/2},
```

where $\alpha$ is `vorticity.random_freq_decay`. The seed is fixed by `vorticity.random_seed`.

The right-hand side is implemented in {py:func}`quickpde.pdes.Vorticity.get_rhs`.

### Default parameters

**Two-bump preset (`-cn vorticity`)**

| Parameter | Value |
|---|---|
| `axis_points` | 256 |
| `dt` | 2e-3 |
| `t_end` | 200 |
| `store_every` | 200 |
| `viscosity` | 0.0 |
| `ic_sharpness` | 5.0 |
| `vorticity.initial` | `twobump` |
| `vorticity.bump_distance` | 2.0 |
| `vorticity.hyperviscosity_mag` | 500 |
| `vorticity.hyperviscosity_exp` | 8 |

**Random-field preset (`-cn vorticity_grf`)**

| Parameter | Value |
|---|---|
| `axis_points` | 256 |
| `dt` | 1e-3 |
| `t_end` | 100 |
| `store_every` | 1000 |
| `use_double_precision` | `True` |
| `vorticity.initial` | `random` |
| `vorticity.random_freq_decay` | 2.0 |
| `vorticity.random_seed` | 0 |

### Running

Two-bump simulation at reduced resolution:

```bash
python quickpde/driver.py -cn vorticity axis_points=128 t_end=50
```

Ensemble of 10 random-field trajectories with different seeds:

```bash
python quickpde/driver.py --multirun -cn vorticity_grf vorticity.random_seed="range(0,10)"
```
