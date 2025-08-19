# Rotation 2d

- Domain: 2d
- Fields: 1
- Properties: Transport, linear
- Modifiers: IC_sharpness

## Explanation

Simple 2d rotation PDE: One field in a 2-dimensional spatial domain $[-\pi,\pi)^2$ with periodic boundary conditions. The field $u: [0,T] \times [-\pi, \pi)^2 \to \R$ is the solution to the PDE
```{math}
:label: eq-rotation

\partial_t u= x_1 \partial_{x_2}u - x_2 \partial_{x_1} u,
```
The default initial condition is $u_0(\boldsymbol{x})=\exp(-80((x_1-1)^2 + x_2^2))$, where $\boldsymbol{x}=[x_1, x_2]$.

The right-hand side of this PDE is implemented in {py:func}`quickpde.pdes.rotation_2d`

The solution just moves the bump in a circle. Trajectory snapshot matrices will have a slow singular value decay, but the solutions are conceptually very simple, so this is a nice example for nonlinear model order reduction methods.

```bash
python quickpde/driver -cn rotation
```

## Citation
```bibtex
@Misc{SchwerdtnerMBFP2025Operator,
    title	= {Operator Inference Aware Quadratic Manifolds with Isotropic Reduced Coordinates for Nonintrusive Model Reduction},
    author	= {Paul Schwerdtner and Prakash Mohan and Julie Bessac and Marc T. Henry de Frahan and Benjamin Peherstorfer},
    year	= {2025},
    archiveprefix={arXiv},
    primaryclass= {math.DS},
    url		= {https://arxiv.org/abs/2507.20463},
    doi		= {10.48550/arXiv.2507.20463}
}
```
