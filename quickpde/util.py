import functools
import logging
import time

import humanize
from omegaconf import OmegaConf
from quickpde.config import Config


def log_duration(func):

  @functools.wraps(func)
  def wrapper(cfg: Config):
    t0 = time.time()
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    result = func(cfg)
    logging.info(
        "Driver complete. Took %s",
        humanize.naturaldelta(time.time() - t0, minimum_unit="microseconds"),
    )
    return result

  return wrapper

