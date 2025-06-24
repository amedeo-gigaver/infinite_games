from httpx import AsyncClient


class TestHealthRoutes:
    async def test_health(self, test_api_client_no_auth: AsyncClient):
        response = await test_api_client_no_auth.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "OK"}
