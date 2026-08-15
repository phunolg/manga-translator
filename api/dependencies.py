from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from module.database.postgre.database import AsyncSessionLocal, db_session_context


async def get_db() -> AsyncSession:
    """Dependency để set session vào context và tự động cleanup"""
    async with AsyncSessionLocal() as session:
        # Set session vào context variable
        token = db_session_context.set(session)
        try:
            yield session
        finally:
            await session.close()
            # Reset context
            db_session_context.reset(token)

