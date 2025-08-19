from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
  tag: str = 'default'
  pde: str = 'rotation_2d'
  domain_dim: int = 2
  # Grid
  axis_points: int = 128
  bound_x: Tuple[float, float] = (0.0, 1.0)
  bound_y: Tuple[float, float] = (0.0, 1.0)
  # Integration
  dt: float = 1e-2
  t_end: float = 1.0
  store_every: int = 1
  # IC
  ic_sharpness: float = 1.0
  # PDE
  viscosity: float = 1e-2
  # RDE
  injection_rate: float = 3.0
