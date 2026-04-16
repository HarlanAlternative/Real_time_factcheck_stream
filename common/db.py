from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from common.schemas import FactCheckResult


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


class FactCheckRecord(Base):
    __tablename__ = "factcheck_results"

    claim_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)


def create_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    return create_async_engine(database_url, echo=echo, future=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def persist_result(
    session_factory: async_sessionmaker[AsyncSession],
    result: FactCheckResult,
) -> None:
    async with session_factory() as session:
        session.add(
            FactCheckRecord(
                claim_id=str(result.id),
                claim_text=result.text,
                claim_timestamp=result.timestamp,
                label=result.label,
                confidence=result.confidence,
                reasoning=result.reasoning,
                model_name=result.model_name,
                processed_at=result.processed_at,
                raw_response=result.raw_response,
            )
        )
        await session.commit()
