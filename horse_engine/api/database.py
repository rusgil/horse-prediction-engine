from contextlib import asynccontextmanager
from horse_engine.models.database import SessionLocal


@asynccontextmanager
async def get_session():
    async with SessionLocal() as session:
        yield session
