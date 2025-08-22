# quickpde Documentation

1. Package for quickly getting some PDE-solutions of simple equations.
2. Minimal abstractions so the PDE structure is not hidden.
3. Using `jax` for speed and GPU acceleration.

Every PDE has default initial conditions that can be modified with unified parameters like `ic_sharpness` to make problems more or less hard.
Moreover, we include an option for random initialization, so we can generate training data quickly.

```{toctree}
:maxdepth: 2
:hidden:

api/index
api/modules
RDE
Rotation2d
Vorticity
```

