from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from src.api.models import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    chats = relationship("Chat", back_populates="users")