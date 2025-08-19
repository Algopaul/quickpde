from hydra.core.config_store import ConfigStore

import quickpde.config.defaults
from quickpde.config.base import Config

cs = ConfigStore.instance()
cs.store(name='config', node=Config)
