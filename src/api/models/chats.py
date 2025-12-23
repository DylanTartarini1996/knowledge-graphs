from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.api.models import Base

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_id = Column(Integer, nullable=True)
    history = Column(JSONB, nullable=False)   # JSON field for message history
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    users = relationship("User", back_populates="chats")