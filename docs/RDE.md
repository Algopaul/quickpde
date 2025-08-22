# Rotating Detonation Waves 1d

- Domain: 1d
- Fields: 2
- Properties: Transport, nonlinear, shock wave
- Modifiers: viscosity, supply_rate

## Explanation

Nonlinear model for rotating detonation waves that travel over a circular domain, spawning multiple waves depending on supply_rate (injection parameter) $\mu$.
The governing equations are formulated over a spatial domain $[0, 2\pi)$ and given by
```{math}
:label: eq-rde
\partial_t \eta &=-\eta\partial_x \eta+\nu \partial_{xx} \eta + (1-\lambda)\omega(\eta) +\xi(\eta),\\
\partial_t \lambda &= \nu \partial_{xx} \lambda + (1-\lambda)\omega(\eta)-\beta(\eta)\lambda,
```
for functions $\lambda, \eta: [0, T] \times [0, 2\pi) \to \R$ with periodic boundary conditions in the spatial coordinate. The fields $\eta$ and $\lambda$ describe the intensive property of the working fluid and the combustion progress, respectively.

The initial conditions are given by[^1]
```{math}
\eta(x, 0) = \frac3{2\cosh(x - 1.0)^2}
```
and $\lambda(x,0)=0$.

### Additional terms

- viscosity $\nu=10^{-4}$
- heat release function: $\omega(\eta(x, t))=\exp(\frac{\eta(x, t)-\eta_C}{\alpha})$, activation energy $\alpha=0.3$, ignition factor $\eta_C=1.1$
- injection term $\beta(\eta(x, t); \mu)=\frac{\mu}{1+\exp(r(\eta(x, t)-\eta_p))}$, where $\eta_p=0.5$, $r=5$, and injection parameter $\mu=3.5$
- energy loss function $\xi(\eta(x,t))=-\epsilon\eta(x, t)$, $\epsilon=0.11$

The right-hand side of this PDE is implemented in {py:func}`quickpde.pdes.RDE1d.get_rhs`

```bash
python quickpde/driver.py -cn rde
```


## Citation

```bibtex
@Article{KochKKK2020Mode-locked,
    title	= {Mode-locked rotating detonation waves: Experiments and a model equation},
    volume	= {101},
    issn	= {2470-0053},
    url		= {http://dx.doi.org/10.1103/PhysRevE.101.013106},
    doi		= {10.1103/physreve.101.013106},
    number	= {1},
    journal	= {Phys. Rev. E},
    publisher	= {American Physical Society (APS)},
    author	= {Koch, James and Kurosaka, Mitsuru and Knowlen, Carl and Kutz, J. Nathan},
    year	= {2020}
}

@Article{SinghUP2023Lookahead,
    author	= {Singh, Rodrigo and Uy, Wayne Isaac Tan and Peherstorfer, Benjamin},
    title	= {Lookahead data-gathering strategies for online adaptive model reduction of transport-dominated problems},
    journal	= {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    volume	= {33},
    number	= {11},
    pages	= {113112},
    year	= {2023},
    issn	= {1054-1500},
    doi		= {10.1063/5.0169392},
    url		= {https://doi.org/10.1063/5.0169392}
}
```

[^1]: In [SinghUP2023Lookahead](#Citation), the initial conditions are given by
$\eta(x,0)=\frac{3}{2(1-\cosh(x-1))^{20}}$, which is infinite for $x=1$, so we do not use this here. Instead, we use the initial condition from [KochKKK2020Mode-locked](#Citation)
