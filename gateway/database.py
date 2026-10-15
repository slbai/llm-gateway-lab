from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, desc, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestMetric(Base):
    __tablename__ = "request_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    estimated_cost_microusd: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class MetricRepository:
    def __init__(self, database_url: str):
        self.engine: AsyncEngine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(self.engine, class_=AsyncSession)

    async def create_tables(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def save(
        self,
        model: str,
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        metric = RequestMetric(
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_microusd=0,
        )
        async with self.sessions.begin() as session:
            session.add(metric)

    async def list_recent(self) -> list[RequestMetric]:
        async with self.sessions() as session:
            result = await session.scalars(select(RequestMetric).order_by(desc(RequestMetric.id)))
            return list(result)
