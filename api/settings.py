import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')

# Connection pool settings
DB_POOL_SIZE = 5  # Số connection tối đa trong pool
DB_MAX_OVERFLOW = 10
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
DB_POOL_PRE_PING = True

origins = [o.strip() for o in os.getenv('BACKEND_CORS_ORIGINS', 'http://localhost:5173, http://localhost:3000').split(',') if o.strip()]

class Settings(BaseSettings):
    DATABASE_URL: str = f'postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    DB_POOL_SIZE: int = DB_POOL_SIZE
    DB_MAX_OVERFLOW: int = DB_MAX_OVERFLOW
    DB_POOL_TIMEOUT: int = DB_POOL_TIMEOUT
    DB_POOL_RECYCLE: int = DB_POOL_RECYCLE
    DB_POOL_PRE_PING: bool = DB_POOL_PRE_PING

settings = Settings()