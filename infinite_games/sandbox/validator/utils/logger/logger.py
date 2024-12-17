import logging
from abc import ABC

from infinite_games.sandbox.validator.utils.logger.context import start_session, start_trace
from infinite_games.sandbox.validator.utils.logger.formatters import ConsoleFormatter, JSONFormatter


# Abstract logger class for typing
class AbstractLogger(logging.Logger, ABC):
    def start_session(self) -> None:
        pass

    def start_trace(self) -> None:
        pass


# Override the default logging.Logger.makeRecord method to keep extra data
def make_record_with_extra(self, *args, **kwargs):
    record = original_makeRecord(self, *args, **kwargs)

    record._extra = args[-2]

    return record


# Replace the original makeRecord method with the custom implementation
original_makeRecord = logging.Logger.makeRecord
logging.Logger.makeRecord = make_record_with_extra


# Factory function to create and configure a logger with multiple handlers
def create_logger(name: str = None, level: any = logging.DEBUG) -> AbstractLogger:
    # Console message handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())

    # Console JSON handler
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(JSONFormatter())

    # Initialize the logger with the specified name and level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(json_handler)

    # Attach the start methods to the logger
    logger.start_session = start_session
    logger.start_trace = start_trace

    return logger


logger = create_logger("validator")
