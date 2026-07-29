import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://postgresadmin:Thangam-admin@rag-postgres-thangarasu.postgres.database.azure.com:5432/role_based_rag?ssl=require"

async def test():
    engine = create_async_engine(DATABASE_URL)
    async with engine.connect() as conn:
        print("Connected!")

asyncio.run(test())