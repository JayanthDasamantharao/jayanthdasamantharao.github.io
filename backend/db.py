from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


def create_db_engine(database_url: str):
    # Render (and other hosts) hand out bare "postgresql://" URLs, which makes
    # SQLAlchemy default to the psycopg2 driver. We only install psycopg (v3),
    # so force that dialect explicitly regardless of how the URL was supplied.
    if database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]

    # Session TZ UTC so timestamptz round-trips as UTC-aware in Python/SQLAlchemy.
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if database_url.startswith("postgresql"):
        kwargs["connect_args"] = {"options": "-c timezone=UTC"}
    return create_engine(database_url, **kwargs)


def create_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
