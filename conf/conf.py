#for logging and dynaconf
import logging
from dynaconf import Dynaconf

logger = logging.basicConfig(level=logging.INFO)

settings = Dynaconf(settings_file="setting.toml")