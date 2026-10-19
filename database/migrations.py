"""Lightweight, documented schema-version mechanism.

``MIGRATIONS`` maps a schema version number to a function that upgrades the
database *to* that version. On start-up :func:`run_migrations` reads the
highest applied version from ``schema_version`` and applies every newer step in
order, each inside its own transaction.

To change the schema later:
1. edit ``database/models.py``;
2. add ``def _migrate_to_2(conn): ...`` issuing the required ``ALTER TABLE``;
3. register it in ``MIGRATIONS`` and bump ``CURRENT_SCHEMA_VERSION``.
"""

from __future__ import annotations

from typing import Callable

from sqlalchemy import Connection, Engine, inspect, select

from database.models import Base, SchemaVersion, utcnow

CURRENT_SCHEMA_VERSION = 1


def _migrate_to_1(conn: Connection) -> None:
    """Initial schema: create all tables declared in ``database.models``."""
    Base.metadata.create_all(conn)


MIGRATIONS: dict[int, tuple[str, Callable[[Connection], None]]] = {
    1: ("Initial MetaOpt-AI schema", _migrate_to_1),
}


def current_version(engine: Engine) -> int:
    if not inspect(engine).has_table(SchemaVersion.__tablename__):
        return 0
    with engine.connect() as conn:
        versions = conn.execute(select(SchemaVersion.version)).scalars().all()
    return max(versions, default=0)


def run_migrations(engine: Engine) -> int:
    """Apply all pending migrations; return the resulting schema version."""
    SchemaVersion.__table__.create(engine, checkfirst=True)
    applied = current_version(engine)
    for version in sorted(MIGRATIONS):
        if version <= applied:
            continue
        description, step = MIGRATIONS[version]
        with engine.begin() as conn:
            step(conn)
            conn.execute(
                SchemaVersion.__table__.insert().values(
                    version=version, description=description, applied_at=utcnow()
                )
            )
        applied = version
    # Defensive: guarantees tables exist even if a file was partially deleted.
    Base.metadata.create_all(engine)
    return applied
