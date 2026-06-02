import sqlite3
from pathlib import Path

from .models import Chunk

_CREATE = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    doc_path    TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT NOT NULL,
    breadcrumb  TEXT NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_CREATE)


def clear_source(db_path: str | Path, source: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))


def save_chunks(chunks: list[Chunk], db_path: str | Path) -> None:
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(id, source, doc_path, url, title, breadcrumb, content, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (c.chunk_id, c.source, c.doc_path, c.url, c.title, c.breadcrumb, c.content, c.token_count)
                for c in chunks
            ],
        )


def load_chunks(db_path: str | Path, source: str | None = None) -> list[Chunk]:
    with _connect(db_path) as conn:
        if source:
            rows = conn.execute("SELECT * FROM chunks WHERE source = ?", (source,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
    return [
        Chunk(
            chunk_id=row["id"],
            source=row["source"],
            doc_path=row["doc_path"],
            url=row["url"],
            title=row["title"],
            breadcrumb=row["breadcrumb"],
            content=row["content"],
            token_count=row["token_count"],
        )
        for row in rows
    ]


def count_chunks(db_path: str | Path) -> dict[str, int]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM chunks GROUP BY source"
        ).fetchall()
    return {row["source"]: row["cnt"] for row in rows}
