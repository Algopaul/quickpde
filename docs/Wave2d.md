# Wave Equation 2d

- Domain: 2d, $[-\pi, \pi)^2$
- Fields: 3 (density ρ, velocities $v_x$, $v_y$)
- Properties: Transport, linear, Hamiltonian
- Modifiers: `ic_sharpness`

## Explanation

Linear acoustic wave equation written as a first-order system. The three fields $\rho, v_x, v_y : [0, T] \times [-\pi, \pi)^2 \to \mathbb{R}$ satisfy

```{math}
:label: eq-wave2d

\partial_t \rho &= -(\partial_x v_x + \partial_y v_y), \\
\partial_t v_x &= -\partial_x \rho, \\
\partial_t v_y &= -\partial_y \rho,
```

with periodic boundary conditions. The system is Hamiltonian with conserved energy $H = \frac{1}{2}\int (\rho^2 + v_x^2 + v_y^2)\,dx\,dy$.

The default initial condition is a Gaussian bump in the density field,

```{math}
\rho_0(\boldsymbol{x}) = \exp(-\alpha(x^2 + y^2)), \quad v_x = v_y = 0,
```

where $\alpha$ is controlled by `ic_sharpness`. The bump expands outward as a circular wave and wraps around the periodic domain.

The right-hand side is implemented in {py:func}`quickpde.pdes.Wave2d.get_rhs`.

### Default parameters

| Parameter | Value |
|---|---|
| `axis_points` | 512 |
| `dt` | 1e-3 |
| `t_end` | 8 |
| `ic_sharpness` | 1.0 |
| `viscosity` | 0.0 |

### Running

```bash
python quickpde/driver.py -cn wave2d
```

Override resolution and run time:

```bash
python quickpde/driver.py -cn wave2d axis_points=256 t_end=4
```

## Citation

```bibtex
@Article{SchwerdtnerSBP2024Nonlinear,
    author	= {Schwerdtner, Paul and Schulze, Philipp and Berman, Jules and Peherstorfer, Benjamin},
    title	= {Nonlinear Embeddings for Conserving Hamiltonians and Other Quantities with Neural Galerkin Schemes},
    journal	= {SIAM Journal on Scientific Computing},
    volume	= {46},
    number	= {5},
    pages	= {C583-C607},
    year	= {2024},
    doi		= {10.1137/23M1607799}
}
```
