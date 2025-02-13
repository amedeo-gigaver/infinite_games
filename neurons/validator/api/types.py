from fastapi import Request

from neurons.validator.db.operations import DatabaseOperations


class ApiRequest(Request):
    class State:
        db_operations: DatabaseOperations

    state: State
