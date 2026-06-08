import asyncio
from sqlalchemy import select
from config.database import AsyncSessionLocal, engine, Base
from models.model import User, Department, UserRole
from app.auth.security import hash_password
import app.db.models  # register all models


# ── Exact users from problem statement ────────────────────────────────────
SEED_USERS = [
    # C-Level — full access to ALL departments
    {
        "email": "tony@finsolve.com", "username": "tony_cio",
        "password": "Tony@Admin1", "full_name": "Tony Sharma (CIO)",
        "department": Department.c_level, "role": UserRole.admin,
    },
    # Finance Team
    {
        "email": "sam@finsolve.com", "username": "sam_finance",
        "password": "Finance@123", "full_name": "Sam Wilson",
        "department": Department.finance, "role": UserRole.manager,
    },
    {
        "email": "bruce@finsolve.com", "username": "bruce_finance",
        "password": "Finance@456", "full_name": "Bruce Banner",
        "department": Department.finance, "role": UserRole.analyst,
    },
    # Marketing Team
    {
        "email": "wanda@finsolve.com", "username": "wanda_marketing",
        "password": "Market@123", "full_name": "Wanda Maximoff",
        "department": Department.marketing, "role": UserRole.manager,
    },
    {
        "email": "vision@finsolve.com", "username": "vision_marketing",
        "password": "Market@456", "full_name": "Vision",
        "department": Department.marketing, "role": UserRole.analyst,
    },
    # HR Team
    {
        "email": "natasha@finsolve.com", "username": "natasha_hr",
        "password": "HrPass@123", "full_name": "Natasha Romanoff",
        "department": Department.hr, "role": UserRole.manager,
    },
    {
        "email": "steve@finsolve.com", "username": "steve_hr",
        "password": "HrPass@456", "full_name": "Steve Rogers",
        "department": Department.hr, "role": UserRole.analyst,
    },
    # Engineering Team
    {
        "email": "peter@finsolve.com", "username": "peter_eng",
        "password": "Engineer@1", "full_name": "Peter Pandey",
        "department": Department.engineering, "role": UserRole.manager,
    },
    {
        "email": "rhodey@finsolve.com", "username": "rhodey_eng",
        "password": "Engineer@2", "full_name": "James Rhodes",
        "department": Department.engineering, "role": UserRole.analyst,
    },
    # Employee Level — general info only
    {
        "email": "employee1@finsolve.com", "username": "john_emp",
        "password": "Employee@1", "full_name": "John Doe",
        "department": Department.general, "role": UserRole.viewer,
    },
    {
        "email": "employee2@finsolve.com", "username": "jane_emp",
        "password": "Employee@2", "full_name": "Jane Smith",
        "department": Department.general, "role": UserRole.viewer,
    },
]


async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created\n")

    async with AsyncSessionLocal() as db:
        for u in SEED_USERS:
            res = await db.execute(select(User).where(User.email == u["email"]))
            if res.scalar_one_or_none():
                print(f"  skip  {u['email']}")
                continue
            db.add(User(
                email=u["email"],
                username=u["username"],
                full_name=u["full_name"],
                hashed_password=hash_password(u["password"]),
                department=u["department"],
                role=u["role"],
                is_active=True,
                is_verified=True,
            ))
            print(f"  ✅ {u['email']:35s} [{u['department'].value:12s} / {u['role'].value}]")
        await db.commit()
    print("\nSeed done ✓")


if __name__ == "__main__":
    asyncio.run(seed())