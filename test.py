import asyncio
from sqlalchemy import text

from config.database import engine


async def test_connection():
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("✅ Database Connected")
            print("Result:", result.scalar())

    except Exception as e:
        print("❌ Database Connection Failed")
        print(e)


if __name__ == "__main__":
    asyncio.run(test_connection())