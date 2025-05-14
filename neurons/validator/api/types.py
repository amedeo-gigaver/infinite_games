from fastapi import Request

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.config import IfgamesEnvType


class ApiRequest(Request):
    class State:
        db_operations: DatabaseOperations
        env: IfgamesEnvType

    state: State
