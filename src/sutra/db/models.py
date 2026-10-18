"""SQLAlchemy models for the tables in ARCHITECTURE §12.

One Postgres holds everything: vectors, full-text search, sessions, memory, cache and feedback
(ADR-002). Schema changes after Phase 2 need explicit sign-off.

Note on `metadata`: SQLAlchemy's declarative base reserves that attribute name, so the Python
attribute is `meta` while the database column keeps the name §12 specifies.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# bge-small-en-v1.5 (ADR-004). Must match Settings.embedding_dims; the migration hardcodes it
# because changing an embedding dimension is a re-index, not a config tweak.
EMBEDDING_DIMS = 384


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )


class Document(Base):
    """One corpus file."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    # # CONCEPT: Content hashing makes ingestion idempotent — re-running `make ingest` on an
    # # unchanged file is a no-op instead of a duplicate.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Chunk(Base):
    """A retrievable passage. Carries both a dense embedding and a sparse tsvector, because
    hybrid retrieval needs both (ARCHITECTURE §4)."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    # # CONCEPT: Index versioning — a re-ingest writes a new version while the old one keeps
    # # serving reads, so eval baselines stay pinned to a known index.
    index_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Parent-child chunking (Phase 3): retrieve on the small child, feed the large parent to
    # the model. NULL means this chunk is itself a parent, or chunking is flat.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True
    )
    chunk_no: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Contextual retrieval (Phase 3): a short generated prefix situating the chunk in its
    # document, prepended before embedding to fight context loss on small chunks.
    context_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMS), nullable=True)
    tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        # A document's chunk numbering is unique per index version.
        UniqueConstraint("document_id", "index_version", "chunk_no", name="uq_chunk_position"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_index_version", "index_version"),
        Index("ix_chunks_parent_id", "parent_id"),
        # # CONCEPT: HNSW is an approximate index — it trades a little recall for a large speed
        # # gain. Cosine ops match bge-small, which produces normalised embeddings.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": "16", "ef_construction": "64"},
        ),
        Index("ix_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
    )


class Session(Base):
    """A conversation."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_active_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_sessions_user_id", "user_id"),)


class Message(Base):
    """One turn. `prompt_version` and `trace_id` are recorded per message so any answer can be
    traced back to the exact prompt that produced it (ADR-006)."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system','tool')", name="ck_message_role"),
        Index("ix_messages_session_id_created_at", "session_id", "created_at"),
        Index("ix_messages_trace_id", "trace_id"),
    )


class SessionSummary(Base):
    """The rolling summary half of short-term memory (ADR-013). `covers_until_message_id` marks
    how far the summary has consumed, so compaction is resumable and never double-counts."""

    __tablename__ = "session_summaries"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    covers_until_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class UserMemory(Base):
    """A durable fact about a user (ADR-013). Embedded so retrieval can pull only the relevant
    facts instead of replaying all of them."""

    __tablename__ = "user_memories"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMS), nullable=True)
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_user_memories_user_id", "user_id"),
        Index(
            "ix_user_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": "16", "ef_construction": "64"},
        ),
    )


class SemanticCache(Base):
    """Semantic cache (ADR-012).

    # CONCEPT: A semantic cache keyed only by query similarity is a correctness bug waiting to
    # happen. The key must also include prompt_version, index_version, model and user — change a
    # prompt or re-index and old entries must stop matching, or you serve a stale answer. Scoping
    # by user_id is what stops one user's answer leaking to another.
    """

    __tablename__ = "semantic_cache"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMS), nullable=True
    )
    answer: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    index_version: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    hits: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The lookup is always scoped by this composite key before similarity is considered.
        Index(
            "ix_semantic_cache_lookup",
            "user_id",
            "prompt_version",
            "index_version",
            "model",
        ),
        Index("ix_semantic_cache_expires_at", "expires_at"),
        Index(
            "ix_semantic_cache_embedding_hnsw",
            "query_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"query_embedding": "vector_cosine_ops"},
            postgresql_with={"m": "16", "ef_construction": "64"},
        ),
    )


class Feedback(Base):
    """User feedback, linked to the trace that produced the answer (Phase 4)."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        Index("ix_feedback_trace_id", "trace_id"),
    )


class DocumentACL(Base):
    """Row-level access control. An empty ACL for a document means public (ARCHITECTURE §12).

    # CONCEPT: Retrieval-time ACL is the guardrail most RAG demos skip. Filtering after
    # retrieval still leaks via citation counts and latency; the filter belongs in the query.
    """

    __tablename__ = "document_acl"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    __table_args__ = (Index("ix_document_acl_user_id", "user_id"),)
