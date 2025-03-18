from fastapi import APIRouter

from neurons.validator.models.api import HealthCheckResponse

router = APIRouter()


@router.get(
    "",
)
def health() -> HealthCheckResponse:
    return {"status": "OK"}
