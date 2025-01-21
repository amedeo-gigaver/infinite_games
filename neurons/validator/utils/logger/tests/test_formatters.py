import json
import logging
import sys

from colorama import Fore, Style

from neurons.validator.utils.env import ENVIRONMENT_VARIABLES
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.context import logger_context
from neurons.validator.utils.logger.formatters import JSONFormatter
from neurons.validator.version import __version__


class TestJsonFormatter:
    def test_json_formatter_basic_log(self):
        formatter = JSONFormatter()

        # Create a log record with INFO level
        record = logging.LogRecord(
            "test_logger", logging.INFO, "test_module", 1, "This is a JSON log message", None, None
        )

        logger_context.set({"context_key": "context_value"})

        # Simulate the context
        formatted_message = formatter.format(record)
        # Escape ANSI
        formatted_escaped_message = formatted_message.replace("\x1b", "\\u001b")

        log_data = json.loads(formatted_escaped_message)

        # Ensure the log data contains the correct fields
        assert "timestamp" in log_data
        assert log_data["level"] == Fore.GREEN + Style.BRIGHT + "INFO" + Style.RESET_ALL
        assert log_data["message"] == "This is a JSON log message"
        assert log_data["context_key"] == "context_value"

    def test_json_formatter_with_extra_data(
        self,
    ):
        formatter = JSONFormatter()

        # Add some extra data to the record
        extra_data = {"key": "value"}

        record = logging.LogRecord(
            "test_logger",
            logging.INFO,
            "test_module",
            1,
            "This is a log with extra data",
            None,
            None,
        )
        record.__dict__["_extra"] = extra_data

        formatted_message = formatter.format(record)
        # Escape ANSI
        formatted_escaped_message = formatted_message.replace("\x1b", "\\u001b")

        log_data = json.loads(formatted_escaped_message)

        # Ensure the extra data is included in the JSON log output
        assert "data" in log_data
        assert log_data["data"] == extra_data

    def test_json_formatter_with_exception(self):
        formatter = JSONFormatter()

        # Create an exception to log
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

            record = logging.LogRecord(
                "test_logger",
                logging.ERROR,
                "test_module",
                1,
                "This is a log with an exception",
                None,
                exc_info,
            )

        formatted_message = formatter.format(record)
        # Escape ANSI
        formatted_escaped_message = formatted_message.replace("\x1b", "\\u001b")

        log_data = json.loads(formatted_escaped_message)

        # Ensure the exception information is included in the JSON log output
        assert "exception" in log_data
        assert "ZeroDivisionError" in log_data["exception"]

    def test_json_formatter_with_version(
        self,
    ):
        formatter = JSONFormatter()

        record = logging.LogRecord(
            "test_logger",
            logging.INFO,
            "test_module",
            1,
            "This is a log",
            None,
            None,
        )

        formatted_message = formatter.format(record)
        # Escape ANSI
        formatted_escaped_message = formatted_message.replace("\x1b", "\\u001b")

        log_data = json.loads(formatted_escaped_message)

        # Ensure the version data is included in the JSON log output
        assert "version" in log_data
        assert log_data["version"] == __version__

        assert "commit_hash" in log_data
        assert log_data["commit_hash"] == commit_short_hash

    def test_json_formatter_inline_logs(
        self,
    ):
        # Mock env variable
        ENVIRONMENT_VARIABLES.INLINE_LOGS = True

        formatter = JSONFormatter()

        # Create a log record

        record = logging.LogRecord(
            "test_logger",
            logging.WARNING,
            "test_module",
            1,
            "This is a JSON log message",
            None,
            None,
        )

        # Format the log record
        formatted_message = formatter.format(record)

        message_json = json.loads(formatted_message)

        # Ensure the logs string is correct
        assert "\n" not in formatted_message

        # Ensure the log data contains the correct fields
        assert "timestamp" in message_json
        assert message_json["level"] == "WARNING"
        assert message_json["message"] == "This is a JSON log message"
