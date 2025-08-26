# Vorticity transport 2d

- Domain: 2d
- Fields: 1
- Properties: Transport, nonlinear, rotations
- Modifiers: viscosity, initial conditions

## Explanation

Spectral solver for vorticity transport equation
```{math}
:label: eq-vorticity
\partial_t \omega = (\boldsymbol{u} \cdot \nabla) \omega+\nu \nabla^2 \omega
```
in the domain $[-\pi, \pi)^2$.
We use a stream-function $\Psi(\boldsymbol{x})$ with $-\Delta \Psi = \omega$, $\partial_y \Psi = u_1$, and $\partial_x \Psi = -u_2$, where $\boldsymbol{u}=[u_1, u_2]^\top$.

We are interested in low/no-viscosity evolutions and thus set add a hyper-viscosity term ($\kappa\nabla^8 \omega$) to remove high frequency noise.
