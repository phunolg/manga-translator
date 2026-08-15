from contextvars import ContextVar
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from api.settings import settings

# Context variable để lưu session trong request context
db_session_context: ContextVar[AsyncSession] = ContextVar("db_session")

# Tạo async engine với connection pool configuration
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    # Connection pool settings
    pool_size=settings.DB_POOL_SIZE,  # Số connection tối đa trong pool
    max_overflow=settings.DB_MAX_OVERFLOW,  # Số connection có thể vượt quá pool_size
    pool_timeout=settings.DB_POOL_TIMEOUT,  # Timeout khi lấy connection (seconds)
    pool_recycle=settings.DB_POOL_RECYCLE,  # Thời gian recycle connection (seconds) - tránh connection cũ
    pool_pre_ping=settings.DB_POOL_PRE_PING,  # Kiểm tra connection trước khi dùng (recommended)
    # Async specific settings
    connect_args={
        "server_settings": {
            "application_name": "magiv2",
        },
        "command_timeout": 60,  # Timeout cho command (seconds)
    },
)

# Tạo async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class AsyncDB:
    """Wrapper để truy cập async session từ context, tương tự fastapi-sqlalchemy"""
    
    @property
    def session(self) -> AsyncSession:
        """Lấy session từ context"""
        session = db_session_context.get(None)
        if session is None:
            raise RuntimeError("No async session found in context. Make sure to use get_db() dependency.")
        return session
    
    async def add(self, instance):
        """Add instance to session"""
        self.session.add(instance)
    
    async def commit(self):
        """Commit session"""
        await self.session.commit()
    
    async def refresh(self, instance):
        """Refresh instance"""
        await self.session.refresh(instance)
    
    async def flush(self):
        """Flush pending changes to the database"""
        await self.session.flush()

    async def execute(self, statement):
        """Execute statement"""
        return await self.session.execute(statement)
    
    async def get(self, model, ident):
        """Get instance by id"""
        return await self.session.get(model, ident)
    
    async def delete(self, instance):
        """Delete instance from session"""
        await self.session.delete(instance)

    async def query(self, *entities):
        """Query entities"""
        from sqlalchemy import select
        if len(entities) == 1:
            return await self.session.execute(select(entities[0]))
        return await self.session.execute(select(*entities))
    
    async def delete_all(self, instances):
        """Delete multiple instances from session"""
        for instance in instances:
            await self.delete(instance)

# Tạo instance để import và sử dụng
db = AsyncDB()
