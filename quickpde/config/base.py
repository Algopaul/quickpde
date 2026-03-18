from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VorticityConfig:
  initial: str = 'random'
  random_freq_decay: float = 2.0
  random_seed: int = 0
  bump_distance: float = 1.0
  bump_angle: float = 0.0
  forcing: bool = False
  hyperviscosity_mag: float = 5e2
  hyperviscosity_exp: int = 8


@dataclass
class Config:
  tag: str = 'default'
  pde: str = 'rotation_2d'
  domain_dim: int = 2
  use_double_precision: bool = False
  outfile: str | None = None
  outdir: str = 'data/results'
  # Grid
  axis_points: int = 128
  bound_x: Tuple[float, float] = (0.0, 1.0)
  bound_y: Tuple[float, float] = (0.0, 1.0)
  # Integration
  dt: float = 1e-2
  t_end: float = 1.0
  store_every: int = 1
  store_type: str = 'f4'
  # IC
  ic_sharpness: float = 1.0
  # PDE
  viscosity: float = 1e-4
  # RDE
  injection_rate: float = 3.0
  vorticity: VorticityConfig = field(default_factory=VorticityConfig)
