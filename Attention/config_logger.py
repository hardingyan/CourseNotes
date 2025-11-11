import os
import sys
from loguru import logger


def config_logger():
    DEFAULT_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    rank = int(os.getenv("RANK", "0"))

    logger.remove()
    logger.configure(extra={"rank": rank})
    CUSTOM_FORMAT = DEFAULT_FORMAT.replace(
        "<level>{level: <8}</level> | ",
        "<level>{level: <8}</level> | rank <cyan>{extra[rank]}</cyan> | ",
    )

    logger.add(sys.stderr, format=CUSTOM_FORMAT)

    configured_logger = logger


config_logger()
