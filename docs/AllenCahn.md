# Allen-Cahn Equation

- Domain: 1d, $[0, 1)$
- Fields: 1
- Properties: Reaction-diffusion, nonlinear
- Modifiers: `allen_cahn.epsilon`

## Explanation

Phase-field model for interface dynamics with a time-dependent potential. The field $u : [0, T] \times [0, 1) \to \mathbb{R}$ satisfies

```{math}
:label: eq-allen-cahn

\partial_t u = \varepsilon\,\partial_{xx} u - a(t, x)\,(u - u^3),
```

with periodic boundary conditions, where $\varepsilon$ is the interface-width parameter and the time-dependent coefficient is

```{math}
a(t, x) = 1.05 + t\sin(2\pi x).
```

The reaction term $f(u) = u - u^3$ drives the solution toward the stable phases $u = \pm 1$, while the diffusion term regularises sharp interfaces. The coefficient $a(t,x)$ modulates the strength of the reaction over time.

The initial condition follows [BrunaPV2022](#citation):

```{math}
u_0(x) = \varphi(x;\, 0.03) - \varphi(x;\, 0.7),
\qquad
\varphi(x;\, b) = \exp\!\left(-w^2\sin^2(\pi(x-b))\right),\quad w = \sqrt{20}.
```

The right-hand side is implemented in {py:func}`quickpde.pdes.AllenCahn.get_rhs`.

### Implementation note

Because the RHS depends explicitly on $t$, time is augmented into the ODE state as an extra scalar and stripped from the output before saving. The stored trajectory therefore has shape `(n_frames, axis_points)`.

### Default parameters

| Parameter | Value |
|---|---|
| `axis_points` | 500 |
| `dt` | 1e-3 |
| `t_end` | 12.0 |
| `store_every` | 60 |
| `allen_cahn.epsilon` | 5e-4 |

### Running

```bash
python quickpde/driver.py -cn allen_cahn
```

Override the interface width:

```bash
python quickpde/driver.py -cn allen_cahn allen_cahn.epsilon=1e-3
```

## Citation

```bibtex
@Article{BrunaPV2022Neural,
    author  = {Joan Bruna and Benjamin Peherstorfer and Eric Vanden-Eijnden},
    title   = {Neural {G}alerkin Scheme with Active Learning for High-Dimensional Evolution Equations},
    journal = {Preprint},
    number  = {arXiv:2203.01360},
    year    = {2022},
    doi     = {10.48550/arXiv.2203.01360}
}
```
