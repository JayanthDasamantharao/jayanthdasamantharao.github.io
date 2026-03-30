from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text

from .db import create_db_engine, create_session_factory
from .models import Base, ChatMessage, ChatSession


def _utc_iso_z(dt: datetime) -> str:
    """Serialize timestamps for JSON/JS: UTC instant with Z suffix (avoids ambiguous local parsing)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    s = dt.isoformat(timespec="milliseconds")
    return s.replace("+00:00", "Z")


class ChatStore:
    def __init__(self, database_url: str) -> None:
        self.engine = create_db_engine(database_url)
        self.SessionLocal = create_session_factory(self.engine)
        Base.metadata.create_all(bind=self.engine)
        # Backward-compatible schema update for existing local DBs.
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS intent_confidence DOUBLE PRECISION"))

    @contextmanager
    def _session_scope(self):
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _resolve_primary_intent(db, chat_session_id: int) -> str:
        latest_user_intent: Optional[str] = db.execute(
            select(ChatMessage.intent)
            .where(
                ChatMessage.chat_session_id == chat_session_id,
                ChatMessage.role == "user",
                ChatMessage.intent.is_not(None),
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_user_intent:
            return latest_user_intent

        latest_any_intent: Optional[str] = db.execute(
            select(ChatMessage.intent)
            .where(
                ChatMessage.chat_session_id == chat_session_id,
                ChatMessage.intent.is_not(None),
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        return latest_any_intent or "unknown"

    @staticmethod
    def _resolve_primary_intent_confidence(db, chat_session_id: int) -> Optional[float]:
        latest_user_confidence: Optional[float] = db.execute(
            select(ChatMessage.intent_confidence)
            .where(
                ChatMessage.chat_session_id == chat_session_id,
                ChatMessage.role == "user",
                ChatMessage.intent_confidence.is_not(None),
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_user_confidence is not None:
            return float(latest_user_confidence)

        latest_any_confidence: Optional[float] = db.execute(
            select(ChatMessage.intent_confidence)
            .where(
                ChatMessage.chat_session_id == chat_session_id,
                ChatMessage.intent_confidence.is_not(None),
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_any_confidence is not None:
            return float(latest_any_confidence)
        return None

    def _get_or_create_chat_session(self, db, session_id: str) -> ChatSession:
        existing = db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        ).scalar_one_or_none()
        if existing:
            return existing
        created = ChatSession(session_id=session_id)
        db.add(created)
        db.flush()
        return created

    def log_message(
        self,
        *,
        session_id: str,
        role: str,
        message_text: str,
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        sources: Optional[List[str]] = None,
        is_fallback: bool = False,
        is_abusive_blocked: bool = False,
    ) -> None:
        with self._session_scope() as db:
            chat_session = self._get_or_create_chat_session(db, session_id)
            row = ChatMessage(
                chat_session_id=chat_session.id,
                role=role,
                message_text=message_text,
                intent=intent,
                intent_confidence=intent_confidence,
                sources_json=sources or [],
                is_fallback=is_fallback,
                is_abusive_blocked=is_abusive_blocked,
            )
            db.add(row)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._session_scope() as db:
            rows = db.execute(
                select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
            ).scalars()
            return [
                {
                    "session_id": row.session_id,
                    "primary_intent": self._resolve_primary_intent(db, row.id),
                    "primary_intent_confidence": self._resolve_primary_intent_confidence(db, row.id),
                    "created_at": _utc_iso_z(row.created_at),
                    "updated_at": _utc_iso_z(row.updated_at),
                }
                for row in rows
            ]

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._session_scope() as db:
            chat_session = db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            ).scalar_one_or_none()
            if not chat_session:
                return []
            rows = db.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_session_id == chat_session.id)
                .order_by(ChatMessage.created_at.asc())
            ).scalars()
            return [
                {
                    "role": row.role,
                    "message_text": row.message_text,
                    "intent": row.intent,
                    "intent_confidence": row.intent_confidence,
                    "sources": row.sources_json or [],
                    "is_fallback": row.is_fallback,
                    "is_abusive_blocked": row.is_abusive_blocked,
                    "created_at": _utc_iso_z(row.created_at),
                }
                for row in rows
            ]
