"""Pending email verification for connect / resume-no-fit flows: send a code, verify, then notify."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .meeting_coordinator import MeetingCoordinator


def _pepper() -> str:
    return (os.getenv("EMAIL_VERIFY_PEPPER") or "dev-change-me-email-verify-pepper").strip()


def _hash_code(code: str) -> str:
    return hashlib.sha256(f"{_pepper()}:{code.strip()}".encode("utf-8")).hexdigest()


def _generate_code() -> str:
    return f"{secrets.randbelow(900000) + 100000:06d}"


class ConnectEmailVerificationStore:
    """File-backed pending verifications (one active per chat session_id)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "connect_email_verification.json"
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text(json.dumps({"sessions": {}}, indent=2), encoding="utf-8")

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        raw = json.dumps(data, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.data_dir), prefix="verify_", suffix=".json")
        try:
            os.write(fd, raw.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _cleanup_expired(self, data: Dict[str, Any]) -> None:
        now = time.time()
        sessions = data.get("sessions") or {}
        expired = [sid for sid, row in sessions.items() if float(row.get("expires_at") or 0) <= now]
        for sid in expired:
            sessions.pop(sid, None)

    def get_pending(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        self._cleanup_expired(data)
        row = (data.get("sessions") or {}).get(session_id)
        if not row:
            return None
        if float(row.get("expires_at") or 0) <= time.time():
            (data.get("sessions") or {}).pop(session_id, None)
            self._save(data)
            return None
        return row

    def start(
        self,
        *,
        session_id: str,
        details: Dict[str, str],
        source_message: str,
    ) -> Tuple[bool, Optional[str]]:
        """Send a verification code; store pending state. Returns (ok, error_message)."""
        code = _generate_code()
        ttl = max(120, int(os.getenv("EMAIL_VERIFY_TTL_SECONDS", "900") or "900"))
        now = time.time()
        row = {
            "email": details.get("email") or "",
            "code_hash": _hash_code(code),
            "expires_at": now + ttl,
            "attempts": 0,
            "details": dict(details),
            "source_message": source_message,
        }
        data = self._load()
        self._cleanup_expired(data)
        data.setdefault("sessions", {})[session_id] = row
        self._save(data)

        name = (details.get("name") or "there").strip() or "there"
        to_email = (details.get("email") or "").strip()
        if not to_email:
            return False, "Missing email address."

        subject = "Your verification code for Jayanth's assistant"
        body = (
            f"Hi {name},\n\n"
            f"Your verification code is: {code}\n\n"
            "Paste this code back in the chat to confirm your email. "
            "It expires in a few minutes.\n\n"
            "If you didn't request this, you can ignore this message.\n\n"
            "— Ada, Jayanth's AI assistant"
        )
        html = MeetingCoordinator._email_shell(
            title="Your verification code",
            subtitle="Confirm your email in the chat",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {name},</p>"
                f"<p style='margin:0 0 12px;'>Your verification code is:</p>"
                f"<p style='margin:0 0 16px;font-size:28px;letter-spacing:4px;font-weight:700;color:#0f172a;'>{code}</p>"
                "<p style='margin:0 0 12px;'>Paste this code back in the chat window. It expires shortly.</p>"
                "<p style='margin:0;color:#64748b;font-size:13px;'>If you didn't request this, you can ignore this email.</p>"
            ),
        )
        sent = MeetingCoordinator._send_email(
            subject=subject,
            body=body,
            to_email=to_email,
            html_body=html,
        )
        if not sent:
            data = self._load()
            (data.get("sessions") or {}).pop(session_id, None)
            self._save(data)
            return False, "Email could not be sent (SMTP not configured). Verification was not started."
        return True, None

    def verify(self, session_id: str, code: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Returns (status, payload) where status is ok|wrong|expired|locked|none."""
        data = self._load()
        self._cleanup_expired(data)
        sessions = data.setdefault("sessions", {})
        row = sessions.get(session_id)
        if not row:
            return "none", None
        if float(row.get("expires_at") or 0) <= time.time():
            sessions.pop(session_id, None)
            self._save(data)
            return "expired", None

        max_attempts = max(3, int(os.getenv("EMAIL_VERIFY_MAX_ATTEMPTS", "8") or "8"))
        if int(row.get("attempts") or 0) >= max_attempts:
            sessions.pop(session_id, None)
            self._save(data)
            return "locked", None

        if secrets.compare_digest(row.get("code_hash") or "", _hash_code(code)):
            payload = {
                "details": row.get("details") or {},
                "source_message": row.get("source_message") or "",
            }
            sessions.pop(session_id, None)
            self._save(data)
            return "ok", payload

        row["attempts"] = int(row.get("attempts") or 0) + 1
        sessions[session_id] = row
        self._save(data)
        return "wrong", None

    def resend(self, session_id: str) -> Tuple[bool, Optional[str]]:
        pending = self.get_pending(session_id)
        if not pending:
            return False, "no_pending"
        details = pending.get("details") or {}
        source_message = pending.get("source_message") or ""
        return self.start(
            session_id=session_id,
            details=details,
            source_message=source_message,
        )


_CODE_IN_MESSAGE = re.compile(r"\b(\d{6})\b")


def extract_six_digit_code(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    if text.isdigit() and len(text) == 6:
        return text
    m = _CODE_IN_MESSAGE.search(text)
    return m.group(1) if m else None


def looks_like_resend_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if "resend" not in lower:
        return False
    return "code" in lower or "email" in lower or len(lower) < 48
