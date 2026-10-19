"""Engine / session management for the SQLite database.

* Foreign keys are enforced with ``PRAGMA foreign_keys=ON`` on every connection
  (SQLite disables them by default).
* :func:`session_scope` provides commit-on-success / rollback-on-error
  transaction handling.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import config

_ENGINES: dict[str, Engine] = {}


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Return a cached engine for ``db_path`` (defaults to ``config.DB_PATH``)."""
    path = Path(db_path or config.DB_PATH).resolve()
    key = str(path)
    if key not in _ENGINES:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{path}", future=True, connect_args={"check_same_thread": False}
        )
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
        _ENGINES[key] = engine
    return _ENGINES[key]


def dispose_engine(db_path: str | Path | None = None) -> None:
    """Dispose and forget a cached engine (used by tests to simulate restarts)."""
    key = str(Path(db_path or config.DB_PATH).resolve())
    engine = _ENGINES.pop(key, None)
    if engine is not None:
        engine.dispose()


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: str | Path | None = None) -> Engine:
    """Create the database file (if needed) and apply pending migrations."""
    from database.migrations import run_migrations

    engine = get_engine(db_path)
    run_migrations(engine)
    return engine
