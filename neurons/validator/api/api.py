from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from uvicorn import Config, Server

from neurons.validator.api.middlewares import (
    APIKeyMiddleware,
    LoggingAndErrorHandlingMiddleware,
    StateMiddleware,
)
from neurons.validator.api.routes.root_router import router as root_router
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.config import IfgamesEnvType
from neurons.validator.utils.logger.logger import set_uvicorn_logger


class API:
    host: str
    port: int
    fast_api: FastAPI | None
    server: Server | None
    db_operations: DatabaseOperations
    env: IfgamesEnvType

    def __init__(
        self,
        host: str,
        port: int,
        db_operations: DatabaseOperations,
        env: IfgamesEnvType,
        api_access_keys: str | None,
    ):
        if not isinstance(host, str):
            raise ValueError("host must be a string")

        if not isinstance(port, int):
            raise ValueError("port must be an integer")

        if not isinstance(db_operations, DatabaseOperations):
            raise ValueError("db_operations must be an instance of DatabaseOperations")

        if api_access_keys is not None and not isinstance(api_access_keys, str):
            raise ValueError("api_access_keys must be a string or None")

        if not isinstance(env, str):
            raise ValueError("env must be a string")

        self.host = host
        self.port = port
        self.fast_api = None
        self.server = None
        self.db_operations = db_operations
        self.env = env
        self.api_access_keys = api_access_keys
        self.api_key_header = "X-API-Key"

    def _get_rate_limiter_key(self, request: Request):
        return request.headers.get(self.api_key_header, "fallback-key")

    def _override_openapi(self, app: FastAPI):
        app.openapi_schema = app.openapi()

        app.openapi_schema["components"]["securitySchemes"] = {
            "APIKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }
        }

        for path, path_value in app.openapi_schema["paths"].items():
            for method in path_value:
                path_value[method]["security"] = [{"APIKeyHeader": []}]

    def create_api(self):
        # Create FastAPI
        fast_api = FastAPI(
            title="SN6 Validator API",
            version="0.1.0",
            redoc_url=None,
            docs_url=None,
            openapi_url="/api/openapi.json",
        )
        # Rate limiter
        limiter = Limiter(key_func=self._get_rate_limiter_key, application_limits=["4/1seconds"])
        fast_api.state.limiter = limiter

        fast_api.add_middleware(StateMiddleware, db_operations=self.db_operations, env=self.env)

        fast_api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        fast_api.add_middleware(SlowAPIMiddleware)

        fast_api.add_middleware(
            APIKeyMiddleware,
            allowed_api_keys=self.api_access_keys,
            api_key_header=self.api_key_header,
        )

        fast_api.add_middleware(
            LoggingAndErrorHandlingMiddleware,
            api_key_header=self.api_key_header,
        )

        fast_api.include_router(root_router, prefix="/api")

        self._override_openapi(app=fast_api)

        return fast_api

    async def start(self):
        if not self.api_access_keys:
            return

        if self.fast_api is not None:
            raise RuntimeError("API already started")

        fast_api = self.create_api()

        config = Config(
            fast_api, host=self.host, port=self.port, access_log=False, use_colors=False
        )

        set_uvicorn_logger()

        self.fast_api = fast_api
        self.server = Server(config)

        await self.server.serve()
