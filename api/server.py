from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.controller.app_router import app_router
from api.handler.exception_handler import exception_handler
from api.schema.api_exception_schema import ApiException
from api.settings import origins

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router
app.include_router(app_router)

app.add_exception_handler(ApiException, exception_handler)
app.add_exception_handler(Exception, exception_handler)
app.add_exception_handler(RequestValidationError, exception_handler)
    
if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8008)
