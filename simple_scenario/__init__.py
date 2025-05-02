from loguru import logger
import sys

logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

from ._check_extra_availability import CR_AVAILABLE, LXD_AVAILABLE
from .ego_configuration import EgoConfiguration
from .from_data_config_error import FromDataConfigError
from .scenario import Scenario
from .vehicle import Vehicle

if CR_AVAILABLE:
    from .lanelet_network_wrapper import LaneletNetworkWrapper
