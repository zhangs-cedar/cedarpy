import time
from cedar.utils.logger import init_logger
from cedar.utils import logger

logger.init_logger()
logger.info("hello")
time.sleep(1)
logger.debug("world")
logger.warning("warning")
