from fastapi import APIRouter

from neurons.validator.api.routes.events import router as events_router
from neurons.validator.api.routes.health import router as health_router

router = APIRouter()

router.include_router(health_router, prefix="/health", include_in_schema=False)
router.include_router(events_router, prefix="/events", include_in_schema=True)
