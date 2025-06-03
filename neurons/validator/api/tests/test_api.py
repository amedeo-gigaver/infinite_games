from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware

from neurons.validator.api.api import API
from neurons.validator.api.middlewares import (
    APIKeyMiddleware,
    LoggingAndErrorHandlingMiddleware,
    StateMiddleware,
)
from neurons.validator.db.operations import DatabaseOperations


class TestAPI:
    @pytest.fixture
    def api(self):
        db_operations = MagicMock(spec=DatabaseOperations)

        api = API(
            host="127.0.0.1",
            port=8000,
            db_operations=db_operations,
            env="dev",
            api_access_keys="key",
        )

        return api

    def test_init_validation(self):
        db_operations = MagicMock(spec=DatabaseOperations)

        # Invalid host
        with pytest.raises(ValueError):
            API(host=123, port=8000, db_operations=db_operations, env="dev", api_access_keys=None)

        # Invalid port
        with pytest.raises(ValueError):
            API(
                host="127.0.0.1",
                port="8000",
                db_operations=db_operations,
                env="dev",
                api_access_keys=None,
            )

        # Invalid db_operations
        with pytest.raises(ValueError):
            API(
                host="127.0.0.1",
                port=8000,
                db_operations="not_db_operations",
                env="dev",
                api_access_keys=None,
            )

        # Invalid api_access_keys
        with pytest.raises(ValueError):
            API(
                host="127.0.0.1",
                port=8000,
                db_operations=db_operations,
                env="dev",
                api_access_keys=123,
            )

        # Invalid env
        with pytest.raises(ValueError):
            API(
                host="127.0.0.1",
                port=8000,
                db_operations=db_operations,
                env=123,
                api_access_keys=None,
            )

    def test_get_rate_limiter_key_default(self, api):
        request = Request(scope={"type": "http", "headers": []})

        assert api._get_rate_limiter_key(request) == "fallback-key"

    def test_get_rate_limiter_key_with_header(self, api):
        api_key = "test_api_key"

        headers = [(b"x-api-key", api_key.encode())]
        request = Request(scope={"type": "http", "headers": headers})

        assert api._get_rate_limiter_key(request) == api_key

    def test_create_api(self):
        db_operations = MagicMock(spec=DatabaseOperations)

        api = API(
            host="127.0.0.1",
            port=8000,
            db_operations=db_operations,
            env="dev",
            api_access_keys="key1,key2",
        )

        app = api.create_api()

        # Assert limiter
        assert hasattr(app.state, "limiter")
        assert isinstance(app.state.limiter, Limiter)

        # Assert middlewares and correct order
        middleware_classes = [mw.cls for mw in app.user_middleware]

        assert middleware_classes == [
            LoggingAndErrorHandlingMiddleware,
            APIKeyMiddleware,
            SlowAPIMiddleware,
            CORSMiddleware,
            StateMiddleware,
        ]

        # All routes prefixed
        routes = [route for route in app.router.routes]
        assert any(r.path.startswith("/api") for r in routes)

        # Assert OpenAPI overrides
        schema = app.openapi_schema

        assert "APIKeyHeader" in schema["components"]["securitySchemes"]

        # Every endpoint method has security
        for path_item in schema["paths"].values():
            for operation in path_item.values():
                assert operation["security"] == [{"APIKeyHeader": []}]

    async def test_start_no_access_keys(self):
        db_operations = MagicMock(spec=DatabaseOperations)

        api = API(
            host="127.0.0.1",
            port=8000,
            db_operations=db_operations,
            env="dev",
            api_access_keys=None,
        )

        result = await api.start()

        assert result is None

    async def test_start_twice_raises(dummy_db):
        db_operations = MagicMock(spec=DatabaseOperations)

        api = API(
            host="127.0.0.1",
            port=8000,
            db_operations=db_operations,
            env="dev",
            api_access_keys="key",
        )

        # Simulate already started
        api.fast_api = FastAPI()

        with pytest.raises(RuntimeError, match="API already started"):
            await api.start()
