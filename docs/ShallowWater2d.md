# Shallow Water Equations 2d

- Domain: 2d, $[-\pi, \pi)^2$
- Fields: 2 (height $h$, velocity potential $\phi$)
- Properties: Transport, nonlinear, Hamiltonian
- Modifiers: `ic_sharpness`

## Explanation

Nonlinear shallow water equations in Hamiltonian form. The height $h$ and velocity potential $\phi: [0, T] \times [-\pi, \pi)^2 \to \mathbb{R}$ satisfy

```{math}
:label: eq-swe2d

\partial_t h &= -\bigl(\partial_x(h\,\partial_x\phi) + \partial_y(h\,\partial_y\phi)\bigr), \\
\partial_t \phi &= -\tfrac{1}{2}(\partial_x\phi)^2 - \tfrac{1}{2}(\partial_y\phi)^2 - h,
```

with periodic boundary conditions. The conserved Hamiltonian is

```{math}
H = \int \Bigl[\tfrac{1}{2}h\bigl((\partial_x\phi)^2 + (\partial_y\phi)^2\bigr) + \tfrac{1}{2}h^2\Bigr]\,dx\,dy.
```

The default initial condition is a perturbed flat surface,

```{math}
h_0(\boldsymbol{x}) = 1 + 0.33\exp(-\alpha(x^2 + y^2)), \quad \phi_0 = 0,
```

where $\alpha$ is controlled by `ic_sharpness`. The initial bump disperses into outward-propagating waves.

The right-hand side is implemented in {py:func}`quickpde.pdes.ShallowWater2d.get_rhs`.

### Default parameters

| Parameter | Value |
|---|---|
| `axis_points` | 256 |
| `dt` | 1e-3 |
| `t_end` | 10 |
| `store_every` | 80 |
| `ic_sharpness` | 2.7 |
| `viscosity` | 0.0 |

### Running

```bash
python quickpde/driver.py -cn swe2d
```

Override sharpness to generate a more concentrated initial condition:

```bash
python quickpde/driver.py -cn swe2d ic_sharpness=5.0 axis_points=128
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

@Article{HesthavenPR2022Rank-adaptive,
    author	= {Hesthaven, Jan S. and Pagliantini, Cecilia and Ripamonti, Nicol\`o},
    doi		= {10.1051/m2an/2022013},
    journal	= {ESAIM: M2AN},
    number	= 2,
    pages	= {617-650},
    title	= {Rank-adaptive structure-preserving model order reduction of {H}amiltonian systems},
    volume	= 56,
    year	= 2022
}
```
