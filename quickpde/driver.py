import logging

import h5py
import hydra

from quickpde.config import Config
from quickpde.initial_conditions import get_initial_condition
from quickpde.odesolve import get_ode_solver
from quickpde.pdes import get_pde
from quickpde.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration
def main(cfg: Config) -> None:
  field = get_initial_condition(cfg)
  rhs = get_pde(cfg)
  solver = get_ode_solver(rhs, cfg)
  trajectory, timepoints = solver(field)
  logging.info('Generated trajectory with shape %s', trajectory.shape)
  with h5py.File(f'data/test_{cfg.injection_rate:.2f}.h5', 'w') as f:
    f.create_dataset('data', data=trajectory)
    f.create_dataset('time', data=timepoints)
  pass


if __name__ == "__main__":
  main()
