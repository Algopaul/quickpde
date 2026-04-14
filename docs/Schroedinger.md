# Schrödinger Equation

- Domain: 1d, $[-10, 10)$
- Fields: 2 (real part $\phi_r$, imaginary part $\phi_i$)
- Properties: Dispersive, linear, Hamiltonian
- Modifiers: —

## Explanation

Linear Schrödinger equation with a quartic potential, written as a real-valued first-order system. The complex wavefunction $\Psi = \phi_r + i\phi_i$ satisfies $i\,\partial_t\Psi = \hat{H}\Psi$, which splits into

```{math}
:label: eq-schroedinger

\partial_t \phi_r &= -\tfrac{1}{2}\partial_{xx}\phi_i + V(x)\,\phi_i, \\
\partial_t \phi_i &= \phantom{-}\tfrac{1}{2}\partial_{xx}\phi_r - V(x)\,\phi_r,
```

with periodic boundary conditions. The potential is

```{math}
V(x) = \alpha_2 x^2 + \alpha_4 x^4, \qquad \alpha_2 = -\tfrac{1}{8},\quad \alpha_4 = \tfrac{1}{64}.
```

The system is Hamiltonian with conserved energy $H = \int\!\left(\tfrac{1}{2}|\partial_x\Psi|^2 + V|\Psi|^2\right)dx$ and conserved $L^2$ norm $\|\Psi\|^2 = \int(|\phi_r|^2 + |\phi_i|^2)\,dx$.

The initial condition is a Gaussian wavepacket centred at $q_l = -2$:

```{math}
\phi_r(x, 0) = \pi^{-1/4}\exp\!\left(-\tfrac{(x - q_l)^2}{2}\right), \qquad \phi_i(x, 0) = 0.
```

The field values are near zero at both ends of the domain, so periodic boundary conditions introduce negligible error. The right-hand side is implemented in {py:func}`quickpde.pdes.Schroedinger.get_rhs`.

### Implementation note

Double precision is required (`use_double_precision=True`) because phase errors in $\Psi$ accumulate over the long integration and are invisible in float32 until the solution is qualitatively wrong. The stored trajectory has shape `(n_frames, 2 * axis_points)`, with the first half of each row being $\phi_r$ and the second half $\phi_i$.

### Default parameters

| Parameter | Value |
|---|---|
| `axis_points` | 500 |
| `dt` | 5e-4 |
| `t_end` | 12.0 |
| `store_every` | 120 |
| `use_double_precision` | `True` |

### Running

```bash
python quickpde/driver.py -cn schroedinger
```

Override resolution:

```bash
python quickpde/driver.py -cn schroedinger axis_points=256
```
