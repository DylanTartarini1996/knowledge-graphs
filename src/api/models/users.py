from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base, relationship

from src.api.models import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    chats = relationship("Chat", back_populates="user")