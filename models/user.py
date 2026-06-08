
from pydantic.v1 import BaseModel


class UserCreate(BaseModel):
    username: str
    email: str
    password: str