import logging

from neurons.validator.utils.logger.context import start_session, start_trace
from neurons.validator.utils.logger.formatters import JSONFormatter


class InfiniteGamesLogger(logging.Logger):
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

    # Add a console handler with JSON formatter
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())

    logger.handlers.clear()
    logger.addHandler(json_handler)

    return logger


logger = create_logger("validator")
db_logger = create_logger("db")
