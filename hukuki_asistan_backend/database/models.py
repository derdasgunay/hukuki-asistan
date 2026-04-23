"""
database/models.py
==================
SQLAlchemy 2.0 ORM model for the 'kararlar' table.

Key design decisions:
- EMBEDDING_DIM is a module-level constant so that switching to a different
  model (e.g., one producing 512-dim vectors) requires changing exactly ONE line.
- B-Tree indexes are declared inline with `index=True`. SQLAlchemy will create
  them automatically when Base.metadata.create_all() is called.
- JSON type is used for 'mentioned_laws' to preserve the list structure from
  the source data without a separate join table (the list is read-only metadata).
"""

from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Date, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING DIMENSION — change this single value when switching AI models.
# BERTurk-Legal (BERTurk-Legal_FULL_seed42_ep2_msl192) produces 768-dim vectors.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM: int = 768


class Base(DeclarativeBase):
    """Base class for all ORM models. Provides metadata registry."""
    pass


class Karar(Base):
    """
    Represents a single legal case (karar) in the database.

    Table: kararlar
    """
    __tablename__ = "kararlar"

    # ── Primary Key ───────────────────────────────────────────────────────────
    # The 'id' field from kararlar.json (e.g., "2018_1055") is used directly
    # as the primary key to prevent duplicate insertions in the ETL pipeline.
    id: Mapped[str] = mapped_column(String, primary_key=True)

    # ── Metadata Columns (indexed for fast WHERE-clause filtering) ────────────
    # B-Tree indexes on these columns let PostgreSQL skip full-table scans when
    # a user filters by court name or subject category.
    mahkeme: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    konu: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )

    # ── Case Identifiers ──────────────────────────────────────────────────────
    esas_no: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    karar_no: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # ── Date Columns ──────────────────────────────────────────────────────────
    dava_tarihi: Mapped[Optional[str]] = mapped_column(Date, nullable=True)
    karar_tarihi: Mapped[Optional[str]] = mapped_column(Date, nullable=True)

    # ── Full Text Fields ──────────────────────────────────────────────────────
    olay_ozeti: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tam_olay: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gerekce: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hukum: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Model-Ready Summaries ─────────────────────────────────────────────────
    summary_for_human: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_for_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── RRL Segments (Reasoning-Rule-Law breakdown) ───────────────────────────
    rrl_facts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rrl_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rrl_verdict: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Structural Features ───────────────────────────────────────────────────
    # Stored as JSON so the list ["Ticaret Kanunu (TTK) 55. madde", ...]
    # is preserved exactly as-is from the source data.
    mentioned_laws: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # ── Vector Embedding ─────────────────────────────────────────────────────
    # pgvector stores this as a native PostgreSQL 'vector(768)' column.
    # The <=> operator (cosine distance) is used for similarity search.
    embedding: Mapped[Optional[list]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Karar id={self.id!r} mahkeme={self.mahkeme!r} konu={self.konu!r}>"
