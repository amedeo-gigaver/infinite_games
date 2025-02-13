from contextvars import ContextVar
from uuid import uuid4

logger_context = ContextVar("logger_context", default=None)


def get_context() -> dict:
    return logger_context.get() or {}


def add_context(context: dict):
    current_context = get_context()

    logger_context.set({**current_context, **context})


def start_session():
    current_context = get_context()

    logger_context.set({**current_context, "session_id": str(uuid4())})


def start_trace():
    current_context = get_context()

    logger_context.set({**current_context, "trace_id": str(uuid4())})
