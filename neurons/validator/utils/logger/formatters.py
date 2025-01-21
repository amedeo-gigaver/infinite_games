import json
import logging

from colorama import Back, Fore, Style

from neurons.validator.utils.env import ENVIRONMENT_VARIABLES
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.context import get_context
from neurons.validator.version import __version__


# A custom logging formatter that outputs logs in JSON format
class JSONFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE,
    }

    def format(self, record):
        """
        Format the log record into a JSON object, including log data and context.
        """
        # Retrieve any contextual data (e.g., trace ID) from the current logger context
        context = get_context()

        # Create a dictionary for the log record
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),  # Timestamp of the log event
            "level": record.levelname,  # Log level (INFO, ERROR, etc.)
            "logger": record.name,  # Logger name
            "message": record.getMessage(),  # The actual log message
            **context,  # Add context information
            "version": __version__,  # Version of the validator
            "commit_hash": commit_short_hash,  # Commit hash of the validator
        }

        # Add extra information to the log
        extra_info = record.__dict__.get("_extra", None)

        if extra_info:
            log_record["data"] = extra_info

        # If there is exception information, format it and add to the log record
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # If inline logs, return the JSON string, un-indented and without colors
        if ENVIRONMENT_VARIABLES.INLINE_LOGS:
            return json.dumps(log_record)

        json_str = json.dumps(log_record, indent=2)

        # Add level color format
        level_color = self.COLORS.get(record.levelname, Fore.BLACK)
        colored_level = level_color + Style.BRIGHT + record.levelname + Style.RESET_ALL

        # Only replace first instance of level name to avoid changing it in the log message
        return json_str.replace(record.levelname, colored_level, 1)
