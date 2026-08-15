from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schema.api_exception_schema import ApiException
from utils.log import setup_logger

logger = setup_logger(__name__)

async def exception_handler(request: Request, exc: ApiException | Exception):
    logger.error("\n===== EXCEPTION CAUGHT =====")
    logger.error(f"Path     : {request.url.path}")
    logger.error(f"Method   : {request.method}")
    logger.error(f"Exception: {exc}")
    logger.error("================================\n")
    if isinstance(exc, ApiException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status_code": exc.status_code,
                "is_successful": False,
                "data": exc.message
            }
        )

    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "status_code": 422,
                "is_successful": False,
                "data": exc.errors()
            }
        )
        
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "is_successful": False,
            "data": "Internal server error"
        }
    )