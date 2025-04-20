"""
Main FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from infinite_forecast.api.core.config import get_api_settings
from infinite_forecast.api.core.logging import setup_logging, get_logger
from infinite_forecast.api.routes import events, predictions, resolver

# Initialize settings and logger
settings = get_api_settings()
setup_logging(
    level=settings.log_config.level,
    log_file=settings.log_config.file,
    rotation=settings.log_config.rotation,
    retention=settings.log_config.retention,
)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request processing time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to add processing time header to response.
    
    Args:
        request: Request object
        call_next: Next middleware function
        
    Returns:
        Response with processing time header
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler.
    
    Args:
        request: Request object
        exc: Exception object
        
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(events.router, prefix="/api/v1", tags=["events"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(resolver.router, prefix="/api/v1", tags=["resolver"])

@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        Welcome message
    """
    return {
        "message": "Welcome to Infinite Forecast API",
        "docs_url": "/docs",
        "version": settings.api_version,
    }

@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "ok"}

# Log application startup
@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")

# Log application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info(f"Shutting down {settings.api_title}")

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "infinite_forecast.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    ) 