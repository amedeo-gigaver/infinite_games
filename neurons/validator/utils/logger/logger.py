import logging

from neurons.validator.utils.logger.context import add_context, start_session, start_trace
from neurons.validator.utils.logger.formatters import JSONFormatter


class InfiniteGamesLogger(logging.Logger):
    @property
    def add_context(self):
        return add_context

    @property
    def start_session(self):
        return start_session

    @property
    def start_trace(self):
        return start_trace


logging.setLoggerClass(InfiniteGamesLogger)


# Override the default logging.Logger.makeRecord method to keep extra data
def make_record_with_extra(self, *args, **kwargs):
    record = original_makeRecord(self, *args, **kwargs)

    record._extra = args[-2]

    return record


# Replace the original makeRecord method with the custom implementation
original_makeRecord = logging.Logger.makeRecord
logging.Logger.makeRecord = make_record_with_extra


# Factory function to create and configure a logger
def create_logger(
    name: str = None,
    level: any = logging.DEBUG,
) -> InfiniteGamesLogger:
    # Initialize the logger with the specified name and level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Add a console handler with JSON formatter
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())

    logger.handlers.clear()
    logger.addHandler(json_handler)

    return logger


def set_bittensor_logger():
    bt_logger = logging.getLogger("bittensor")
    bt_logger.propagate = False

    # Add a console handler with JSON formatter
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())

    bt_logger.handlers.clear()
    bt_logger.addHandler(json_handler)

    return bt_logger


def set_uvicorn_logger():
    uvicorn_logger = logging.getLogger("uvicorn")

    # Add a console handler with JSON formatter
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())

    uvicorn_logger.handlers.clear()
    uvicorn_logger.addHandler(json_handler)

    return uvicorn_logger


logger = create_logger("validator")
miner_logger = create_logger("miner")
api_logger = create_logger("api")
