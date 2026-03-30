from __future__ import annotations

import html
import json
import os
import re
import secrets
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

# Jayanth is NJ / US Eastern — slot strings should always show Eastern time.
_EASTERN_TZ_RE = re.compile(r"\b(EST|EDT|ET|Eastern(\s+Time)?)\b", re.I)
_OTHER_TZ_RE = re.compile(
    r"\b(UTC|GMT|IST|CET|PST|CST|MST|PDT|MDT|CDT|JST|AEST|BST)\b",
    re.I,
)


def ensure_est_in_slot_text(text: str) -> str:
    """If the slot has no timezone, append EST (Eastern); leave alone if any tz is already given."""
    t = (text or "").strip()
    if not t:
        return t
    if _EASTERN_TZ_RE.search(t) or _OTHER_TZ_RE.search(t):
        return t
    return f"{t} EST"


@dataclass
class MeetingRequest:
    request_id: str
    token: str
    response_token: str
    name: str
    profession: str
    email: str
    preferred_time: str
    proposed_time: Optional[str]
    proposed_note: Optional[str]
    meeting_link: Optional[str]
    source_message: str
    created_at: str
    status: str = "pending"
    approved_at: Optional[str] = None
    requester_ack_sent: bool = False
    proposal_sent: bool = False
    requester_response: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token": self.token,
            "response_token": self.response_token,
            "name": self.name,
            "profession": self.profession,
            "email": self.email,
            "preferred_time": self.preferred_time,
            "proposed_time": self.proposed_time,
            "proposed_note": self.proposed_note,
            "meeting_link": self.meeting_link,
            "source_message": self.source_message,
            "created_at": self.created_at,
            "status": self.status,
            "approved_at": self.approved_at,
            "requester_ack_sent": self.requester_ack_sent,
            "proposal_sent": self.proposal_sent,
            "requester_response": self.requester_response,
        }


class MeetingCoordinator:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.data_dir / "meeting_requests.json"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        if not self.storage_path.exists():
            self.storage_path.write_text(json.dumps({"requests": []}), encoding="utf-8")

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def missing_fields(details: Dict[str, Optional[str]]) -> List[str]:
        required = ("name", "profession", "email", "preferred_time")
        return [field for field in required if not details.get(field)]

    @staticmethod
    def _send_email(subject: str, body: str, to_email: str, html_body: Optional[str] = None) -> bool:
        host = os.getenv("SMTP_HOST", "").strip()
        username = os.getenv("SMTP_USERNAME", "").strip()
        password = os.getenv("SMTP_PASSWORD", "").strip()
        from_email = os.getenv("SMTP_FROM_EMAIL", "").strip() or username
        port = int(os.getenv("SMTP_PORT", "587"))
        use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes"}

        if not host or not username or not password or not from_email:
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.attach(MIMEText(body, "plain", "utf-8"))
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(host, port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        return True

    @staticmethod
    def _email_shell(*, title: str, subtitle: str, body_html: str) -> str:
        return f"""
        <html>
          <body style="margin:0;padding:0;background:#f5f7fb;font-family:Arial,sans-serif;color:#1f2a44;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="padding:20px 0;">
              <tr>
                <td align="center">
                  <table role="presentation" width="640" cellpadding="0" cellspacing="0" style="max-width:640px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e9f2;">
                    <tr>
                      <td style="background:#0f172a;padding:18px 22px;">
                        <div style="font-size:20px;line-height:1.2;color:#ffffff;font-weight:700;">{title}</div>
                        <div style="margin-top:6px;font-size:13px;color:#cbd5e1;">{subtitle}</div>
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:22px;">
                        {body_html}
                      </td>
                    </tr>
                    <tr>
                      <td style="background:#f8fafc;padding:14px 22px;font-size:12px;color:#64748b;">
                        Sent by Ada, Jayanth's AI assistant
                      </td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>
          </body>
        </html>
        """

    # Statuses where the requester can still cancel or ask to reschedule via chat.
    _ACTIVE_FOR_CHANGE = frozenset(
        {"pending", "awaiting_host_proposal", "proposed", "confirmed", "reschedule_requested"}
    )

    def create_request(
        self,
        details: Dict[str, str],
        source_message: str,
        *,
        chat_session_id: Optional[str] = None,
    ) -> MeetingRequest:
        created_at = datetime.now(timezone.utc).isoformat()
        request = MeetingRequest(
            request_id=secrets.token_hex(8),
            token=secrets.token_urlsafe(24),
            response_token=secrets.token_urlsafe(24),
            name=details["name"],
            profession=details["profession"],
            email=details["email"],
            preferred_time=details["preferred_time"],
            proposed_time=None,
            proposed_note=None,
            meeting_link=None,
            source_message=source_message,
            created_at=created_at,
        )
        payload = self._load()
        row = request.as_dict()
        if chat_session_id:
            row["chat_session_id"] = chat_session_id
        payload["requests"].append(row)
        self._save(payload)
        return request

    def find_latest_active_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Most recent active meeting row for this requester email (e.g. returning after days in a new browser)."""
        em = (email or "").strip().lower()
        if not em:
            return None
        payload = self._load()
        for entry in reversed(payload.get("requests") or []):
            if (entry.get("email") or "").strip().lower() != em:
                continue
            if (entry.get("status") or "") in self._ACTIVE_FOR_CHANGE:
                return dict(entry)
        return None

    def attach_chat_session_to_entry(self, request_id: str, chat_session_id: str) -> Optional[Dict[str, Any]]:
        """Link a new browser/chat session to an existing meeting (after re-verifying the same email)."""
        if not request_id or not chat_session_id:
            return None
        payload = self._load()
        for entry in payload.get("requests") or []:
            if entry.get("request_id") != request_id:
                continue
            entry["chat_session_id"] = chat_session_id
            hist = entry.get("linked_chat_session_ids")
            if not isinstance(hist, list):
                hist = []
            if chat_session_id not in hist:
                hist.append(chat_session_id)
            entry["linked_chat_session_ids"] = hist
            self._save(payload)
            return dict(entry)
        return None

    def create_request_or_link_existing_session(
        self,
        details: Dict[str, str],
        source_message: str,
        chat_session_id: str,
    ) -> Dict[str, Any]:
        """
        If an active meeting already exists for this email, attach chat_session_id and return linked entry.
        Otherwise create a new request (same as create_request).
        """
        email = (details.get("email") or "").strip().lower()
        if email:
            existing = self.find_latest_active_by_email(email)
            if existing:
                upd = self.attach_chat_session_to_entry(existing["request_id"], chat_session_id)
                return {"kind": "linked", "entry": upd if upd else existing}
        req = self.create_request(details, source_message, chat_session_id=chat_session_id)
        return {"kind": "new", "request": req}

    def _build_meeting_link(self) -> str:
        # If configured, use your preferred reusable Google Meet URL.
        configured = os.getenv("DEFAULT_GMEET_LINK", "").strip()
        if configured:
            return configured
        # Fallback: generate-on-open Google Meet link.
        return "https://meet.google.com/new"

    @staticmethod
    def is_placeholder_meet_link(url: Optional[str]) -> bool:
        """True if URL is empty or the generic 'create a new Meet' link (not a stable room)."""
        u = (url or "").strip().lower()
        if not u:
            return True
        return u.rstrip("/") == "https://meet.google.com/new" or u.startswith("https://meet.google.com/new?")

    def notify_jayanth_session_relinked(
        self,
        entry: Dict[str, Any],
        verification_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Email Jayanth when a requester re-verifies and the chat is linked to an existing meeting."""
        notify_email = self._notify_email_default()
        if not notify_email:
            return False
        name = entry.get("name") or "Requester"
        req_email = (entry.get("email") or "").strip()
        status = (entry.get("status") or "").strip()
        proposed = (entry.get("proposed_time") or "").strip()
        preferred = (entry.get("preferred_time") or "").strip()
        when = ensure_est_in_slot_text((proposed or preferred or "").strip() or "see proposal / email")
        vd = verification_details or {}
        vpref_raw = (vd.get("preferred_time") or "").strip()
        vpref = ensure_est_in_slot_text(vpref_raw) if vpref_raw else ""
        mismatch_note = ""
        if vpref and vpref.lower() not in (when.lower(), preferred.lower(), (proposed or "").lower()):
            mismatch_note = (
                f"\n\nThey mentioned different availability in this chat session: {vpref} "
                f"(on-file time is: {when})."
            )
        subject = f"Ada: {name} reconnected — meeting on file ({status})"
        body = (
            f"Hi Jayanth,\n\n"
            f"{name} ({req_email}) just verified their email again and reconnected the Ada chat to their existing meeting.\n\n"
            f"Status on file: {status}\n"
            f"Time on file (authoritative): {when}\n"
            f"{mismatch_note}\n"
            "They can cancel, reschedule, or leave you a note from this chat.\n\n"
            "— Ada, Jayanth's AI assistant"
        )
        mismatch_html = ""
        if vpref and mismatch_note:
            mismatch_html = (
                f"<p style='margin:12px 0 0;padding:12px;background:#fffbeb;border-radius:8px;font-size:14px;'>"
                f"<strong>Latest chat mention:</strong> {html.escape(vpref)} "
                f"(on-file time: {html.escape(when)})</p>"
            )
        html_body = self._email_shell(
            title="Chat reconnected",
            subtitle=f"{name} — same meeting on file",
            body_html=(
                f"<p style='margin:0 0 10px;'><strong>Requester:</strong> {html.escape(name)} — "
                f"<a href='mailto:{html.escape(req_email)}'>{html.escape(req_email)}</a></p>"
                f"<p style='margin:0 0 10px;'><strong>Status:</strong> {html.escape(status)}</p>"
                f"<p style='margin:0 0 14px;'><strong>Time on file:</strong> {html.escape(when)}</p>"
                + mismatch_html
                + "<p style='margin:0;'>They re-verified; this chat is linked to their meeting record.</p>"
            ),
        )
        return self._send_email(subject, body, notify_email, html_body)

    def send_requester_relink_ack(
        self,
        entry: Dict[str, Any],
        verification_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Brief email to the requester: chat reconnected, Jayanth notified, time on file."""
        to = (entry.get("email") or "").strip()
        if not to:
            return False
        name = entry.get("name") or "there"
        status = (entry.get("status") or "").strip()
        proposed = (entry.get("proposed_time") or "").strip()
        preferred = (entry.get("preferred_time") or "").strip()
        when = ensure_est_in_slot_text((proposed or preferred or "").strip() or "see your earlier emails")
        subject = "Your Ada chat is reconnected ✅"
        body = (
            f"Hi {name},\n\n"
            "You’ve verified your email again — this chat is now linked to your existing meeting with Jayanth.\n\n"
            f"Time on file: {when}\n"
            f"Status: {status}\n\n"
            "Jayanth has been notified by email. You can cancel, reschedule, or add a note right here in Ada.\n\n"
            "— Ada, Jayanth's AI assistant"
        )
        html_body = self._email_shell(
            title="Chat reconnected",
            subtitle="Same meeting on file",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {html.escape(name)},</p>"
                "<p style='margin:0 0 12px;'>You’ve verified again — this chat is linked to your meeting with Jayanth.</p>"
                f"<p style='margin:0 0 8px;'><strong>Time on file:</strong> {html.escape(when)}</p>"
                f"<p style='margin:0 0 14px;'><strong>Status:</strong> {html.escape(status)}</p>"
                "<p style='margin:0;'>Jayanth has been emailed a quick heads-up. You can manage this meeting in Ada anytime.</p>"
            ),
        )
        return self._send_email(subject, body, to, html_body)

    def send_requester_ack(self, request: MeetingRequest) -> bool:
        subject = "Thanks for reaching out to Jayanth 🙌"
        body = (
            f"Hi {request.name},\n\n"
            "Thanks for reaching out to connect with Jayanth — really appreciate it.\n"
            "Jayanth will get back to you shortly with a proposed time slot.\n\n"
            "In the meantime, you can explore his LinkedIn profile here:\n"
            "https://www.linkedin.com/in/djayanth/\n\n"
            "Best,\n"
            "Ada, Jayanth's AI assistant"
        )
        html_body = self._email_shell(
            title="Thanks for reaching out 🙌",
            subtitle="Connection request received",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {request.name},</p>"
                "<p style='margin:0 0 12px;'>Thanks for reaching out to connect with Jayanth — really appreciate it. "
                "Jayanth will get back to you shortly with a proposed time slot.</p>"
                "<p style='margin:0 0 12px;'>In the meantime, feel free to explore his LinkedIn profile:</p>"
                "<p style='margin:0 0 12px;'><a href='https://www.linkedin.com/in/djayanth/' "
                "style='color:#1d4ed8;text-decoration:underline;'>https://www.linkedin.com/in/djayanth/</a></p>"
                "<p style='margin:16px 0 0;'>Best,<br/>Ada, Jayanth's AI assistant</p>"
            ),
        )
        return self._send_email(subject=subject, body=body, to_email=request.email, html_body=html_body)

    def notify_jayanth_for_slot_selection(self, request: MeetingRequest) -> bool:
        notify_email = os.getenv("JAYANTH_NOTIFY_EMAIL", "jayanthdasamantharao@gmail.com").strip()
        base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip().rstrip("/")
        propose_url = f"{base_url}/api/meeting/propose/{request.token}"
        subject = f"Meeting Request: {request.name} ({request.profession})"
        body = (
            f"{request.name} ({request.profession}) wants to connect with you.\n\n"
            f"Requester email: {request.email}\n"
            f"Preferred time: {request.preferred_time}\n\n"
            "Propose times in US Eastern (EST/ET) — Jayanth's timezone (NJ).\n"
            "Choose a proposed slot and send it to requester here:\n"
            f"{propose_url}\n"
        )
        html_body = self._email_shell(
            title="New meeting request",
            subtitle=f"{request.name} ({request.profession}) wants to connect",
            body_html=(
                f"<p style='margin:0 0 10px;'><strong>Requester:</strong> {request.name} ({request.profession})</p>"
                f"<p style='margin:0 0 10px;'><strong>Email:</strong> <a href='mailto:{request.email}' "
                f"style='color:#1d4ed8;text-decoration:underline;'>{request.email}</a></p>"
                f"<p style='margin:0 0 14px;'><strong>Preferred time:</strong> {request.preferred_time}</p>"
                "<p style='margin:0 0 14px;font-size:13px;color:#64748b;'>"
                "Use <strong>US Eastern (EST/ET)</strong> in every slot you propose (NJ).</p>"
                f"<p style='margin:0 0 14px;'>Choose a slot and send proposal:</p>"
                f"<p style='margin:0;'><a href='{propose_url}' "
                "style='display:inline-block;background:#1d4ed8;color:#fff;text-decoration:none;padding:10px 14px;border-radius:8px;'>"
                "Open Slot Proposal Form</a></p>"
            ),
        )
        return self._send_email(subject=subject, body=body, to_email=notify_email, html_body=html_body)

    def _find_request_by_token(self, token: str) -> Dict[str, Any]:
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("token") == token:
                return {"payload": payload, "entry": entry}
        return {"payload": payload, "entry": None}

    @staticmethod
    def _normalize_alternative_slots(slots: Optional[List[str]], max_slots: int = 3) -> List[str]:
        out: List[str] = []
        if not slots:
            return out
        for raw in slots[:max_slots]:
            text = (raw or "").strip()
            if text:
                out.append(text)
        return out

    def propose_slot(
        self,
        token: str,
        proposed_time: str,
        note: str = "",
        alternative_slots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        found = self._find_request_by_token(token)
        payload = found["payload"]
        entry = found["entry"]
        if not entry:
            return {"status": "not_found"}

        st = (entry.get("status") or "pending").strip()
        if st not in ("pending", "awaiting_host_proposal", "reschedule_requested"):
            return {"status": "invalid_state", "reason": f"Cannot propose in status={st!r}"}

        base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip().rstrip("/")
        # New token each proposal so old Yes/Need-another-slot links from prior emails are invalidated.
        response_token = secrets.token_urlsafe(24)
        accept_url = f"{base_url}/api/meeting/respond/{response_token}?decision=accept"
        counter_url = f"{base_url}/api/meeting/counter/{response_token}"

        proposed_time = ensure_est_in_slot_text(proposed_time.strip())
        alts_raw = self._normalize_alternative_slots(alternative_slots)
        alts = [ensure_est_in_slot_text(s) for s in alts_raw]

        entry["response_token"] = response_token
        entry["proposed_time"] = proposed_time
        entry["proposed_note"] = note.strip() or None
        entry["alternative_slots"] = alts
        entry["status"] = "proposed"
        entry["proposal_sent"] = True
        self._save(payload)

        subject = "Proposed meeting time with Jayanth"
        note_line = f"\nAdditional note from Jayanth: {entry['proposed_note']}\n" if entry.get("proposed_note") else ""
        body = (
            f"Hi {entry.get('name')},\n\n"
            "Great news — Jayanth reviewed your request and proposed this time "
            "(US Eastern / EST — Jayanth is in NJ):\n"
            f"{proposed_time}\n"
            f"{note_line}\n"
            "Does this time work for you?\n\n"
            f"Yes, this works: {accept_url}\n"
            f"Need another slot (pick alternatives or suggest your own): {counter_url}\n\n"
            "Best,\n"
            "Ada, Jayanth's AI assistant"
        )
        alt_html = ""
        if alts:
            alt_html = "<p style='margin:0 0 8px;'><strong>Alternatives Jayanth also listed:</strong></p><ul style='margin:0 0 12px;padding-left:20px;'>"
            for a in alts:
                alt_html += f"<li style='margin:4px 0;'>{html.escape(a)}</li>"
            alt_html += "</ul>"
        html_body = self._email_shell(
            title="Proposed meeting time",
            subtitle="Please confirm if this works for you",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {entry.get('name')},</p>"
                "<p style='margin:0 0 10px;'>Great news — Jayanth reviewed your request and proposed this slot "
                "(times are <strong>US Eastern (EST/ET)</strong> — Jayanth is in NJ):</p>"
                f"<p style='margin:0 0 12px;'><strong>{html.escape(proposed_time)}</strong></p>"
                + (
                    f"<p style='margin:0 0 12px;'><strong>Note:</strong> {entry.get('proposed_note')}</p>"
                    if entry.get("proposed_note")
                    else ""
                )
                + alt_html
                + "<p style='margin:0 0 14px;'>Does this time work for you?</p>"
                + f"<p style='margin:0 0 8px;'><a href='{accept_url}' "
                "style='display:inline-block;background:#16a34a;color:#fff;text-decoration:none;padding:9px 13px;border-radius:8px;'>"
                "Yes, this works</a></p>"
                + f"<p style='margin:0;'><a href='{counter_url}' "
                "style='display:inline-block;background:#dc2626;color:#fff;text-decoration:none;padding:9px 13px;border-radius:8px;'>"
                "Need another slot</a></p>"
            ),
        )
        sent = self._send_email(subject=subject, body=body, to_email=entry.get("email", ""), html_body=html_body)
        return {"status": "proposed", "request": entry, "proposal_email_sent": sent}

    def _notify_jayanth_counter(self, entry: Dict[str, Any]) -> bool:
        notify_email = os.getenv("JAYANTH_NOTIFY_EMAIL", "jayanthdasamantharao@gmail.com").strip()
        base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip().rstrip("/")
        token = entry.get("token") or ""
        propose_url = f"{base_url}/api/meeting/propose/{token}"
        name = entry.get("name", "Requester")
        counter_msg = entry.get("requester_counter_message") or ""
        subject = f"Requester needs another slot: {name}"
        body = (
            f"{name} ({entry.get('profession')}) needs a different time than the last proposal.\n\n"
            f"What they chose / typed:\n{counter_msg}\n\n"
            f"Requester email: {entry.get('email')}\n"
            f"Originally preferred: {entry.get('preferred_time')}\n"
            f"Last proposed time was: {entry.get('proposed_time')}\n\n"
            "Send a new proposal here:\n"
            f"{propose_url}\n"
        )
        html_body = self._email_shell(
            title="Another slot requested",
            subtitle=f"{name} replied with availability feedback",
            body_html=(
                f"<p style='margin:0 0 10px;'><strong>They said:</strong></p>"
                f"<p style='margin:0 0 14px;padding:12px;background:#f1f5f9;border-radius:8px;'>{html.escape(counter_msg)}</p>"
                f"<p style='margin:0 0 8px;'><strong>Email:</strong> {html.escape(str(entry.get('email') or ''))}</p>"
                f"<p style='margin:0 0 14px;'><strong>Last proposed:</strong> {html.escape(str(entry.get('proposed_time') or ''))}</p>"
                f"<p style='margin:0;'><a href='{propose_url}' "
                "style='display:inline-block;background:#1d4ed8;color:#fff;text-decoration:none;padding:10px 14px;border-radius:8px;'>"
                "Open proposal form again</a></p>"
            ),
        )
        return self._send_email(subject=subject, body=body, to_email=notify_email, html_body=html_body)

    def submit_requester_counter(
        self,
        response_token: str,
        choice: str,
        custom_text: str = "",
    ) -> Dict[str, Any]:
        """Requester picks an alternative Jayanth listed or types their own; notifies Jayanth to propose again."""
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("response_token") != response_token:
                continue
            if entry.get("status") != "proposed":
                return {"status": "invalid_state", "reason": "not awaiting requester response"}

            alts = entry.get("alternative_slots") or []
            choice_norm = (choice or "").strip().lower()
            msg = ""

            if choice_norm.startswith("alt_"):
                try:
                    idx = int(choice_norm.split("_", 1)[1])
                except ValueError:
                    return {"status": "invalid_choice"}
                if idx < 0 or idx >= len(alts):
                    return {"status": "invalid_choice"}
                msg = f"They selected your alternative slot: {alts[idx]}"
            elif choice_norm == "custom":
                text = (custom_text or "").strip()
                if len(text) < 2:
                    return {"status": "invalid_custom", "reason": "Please enter at least a short time or availability."}
                msg = f"They typed their preferred availability: {text}"
            else:
                return {"status": "invalid_choice"}

            entry["status"] = "awaiting_host_proposal"
            entry["requester_response"] = "counter"
            entry["requester_counter_message"] = msg
            hist = entry.get("counter_history") or []
            hist.append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "message": msg,
                }
            )
            entry["counter_history"] = hist
            self._save(payload)
            notified = self._notify_jayanth_counter(entry)
            return {"status": "ok", "request": entry, "jayanth_notified": notified}

        return {"status": "not_found"}

    def get_entry_for_counter_page(self, response_token: str) -> Optional[Dict[str, Any]]:
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("response_token") == response_token:
                return dict(entry)
        return None

    def respond_to_proposal(self, response_token: str, decision: str) -> Dict[str, Any]:
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("response_token") != response_token:
                continue

            decision_norm = (decision or "").strip().lower()
            if decision_norm != "accept":
                return {"status": "invalid_decision"}

            if entry.get("status") != "proposed":
                return {"status": "invalid_state", "reason": "No active proposal to confirm."}

            # accepted
            entry["status"] = "confirmed"
            entry["requester_response"] = "accept"
            entry["approved_at"] = datetime.now(timezone.utc).isoformat()
            meet_link = entry.get("meeting_link") or self._build_meeting_link()
            entry["meeting_link"] = meet_link
            self._save(payload)

            subject = "Meeting confirmed with Jayanth ✅"
            body = (
                f"Hi {entry.get('name')},\n\n"
                "Awesome — your meeting with Jayanth is confirmed.\n"
                f"Scheduled time: {entry.get('proposed_time') or entry.get('preferred_time')}\n"
                f"Google Meet: {meet_link}\n\n"
                "Looking forward to the conversation!\n\n"
                "Best,\n"
                "Ada, Jayanth's AI assistant"
            )
            html_body = self._email_shell(
                title="Meeting confirmed ✅",
                subtitle="Your call with Jayanth is set",
                body_html=(
                    f"<p style='margin:0 0 12px;'>Hi {entry.get('name')},</p>"
                    "<p style='margin:0 0 10px;'>Awesome — your meeting with Jayanth is confirmed.</p>"
                    f"<p style='margin:0 0 10px;'><strong>Scheduled time:</strong> {entry.get('proposed_time') or entry.get('preferred_time')}</p>"
                    f"<p style='margin:0 0 14px;'><strong>Google Meet:</strong> "
                    f"<a href='{meet_link}' style='color:#1d4ed8;text-decoration:underline;'>{meet_link}</a></p>"
                    "<p style='margin:0;'>Looking forward to the conversation!</p>"
                ),
            )
            requester_email = (entry.get("email") or "").strip()
            sent_requester = self._send_email(
                subject=subject, body=body, to_email=requester_email, html_body=html_body
            )
            notify_email = os.getenv("JAYANTH_NOTIFY_EMAIL", "jayanthdasamantharao@gmail.com").strip()
            req_lower = requester_email.lower()
            host_sent = False
            if notify_email:
                if req_lower == notify_email.lower():
                    # Same inbox as requester — one email already contains the confirmation.
                    host_sent = bool(sent_requester)
                else:
                    # Jayanth gets the same confirmation (Meet link + time) as the requester.
                    host_sent = self._send_email(
                        subject=subject, body=body, to_email=notify_email, html_body=html_body
                    )
            entry["confirmation_sent"] = sent_requester
            entry["host_confirmation_sent"] = host_sent
            self._save(payload)
            return {
                "status": "confirmed",
                "request": entry,
                "confirmation_sent": sent_requester,
                "host_confirmation_sent": host_sent,
            }

        return {"status": "not_found"}

    def find_active_request_for_chat(
        self,
        chat_session_id: str,
        requester_email: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Latest active meeting for this chat session, or fallback match by requester email."""
        payload = self._load()
        requests = payload.get("requests") or []
        # Prefer same browser session (most recent last in file — iterate reversed).
        for entry in reversed(requests):
            if entry.get("chat_session_id") != chat_session_id:
                continue
            if (entry.get("status") or "") in self._ACTIVE_FOR_CHANGE:
                return dict(entry)
        if requester_email:
            found = self.find_latest_active_by_email(requester_email)
            if found:
                return found
        return None

    def _notify_email_default(self) -> str:
        return os.getenv("JAYANTH_NOTIFY_EMAIL", "jayanthdasamantharao@gmail.com").strip()

    def _send_cancellation_emails(self, entry: Dict[str, Any]) -> Dict[str, bool]:
        """Polite cancellation notice to requester and Jayanth."""
        name = entry.get("name") or "there"
        req_email = (entry.get("email") or "").strip()
        notify_email = self._notify_email_default()
        when = entry.get("proposed_time") or entry.get("preferred_time") or "your requested time"
        note = (entry.get("cancellation_note") or "").strip()
        note_plain = f"\n\nThey also asked us to share this note:\n{note}\n" if note else ""
        note_html = (
            f"<p style='margin:12px 0 0;padding:12px;background:#f8fafc;border-radius:8px;font-size:14px;'>"
            f"<strong>Note they shared:</strong><br/>{html.escape(note)}</p>"
            if note
            else ""
        )

        sub_r = "Update: your meeting with Jayanth has been cancelled"
        body_r = (
            f"Hi {name},\n\n"
            "Thank you for letting us know. We’ve cancelled the meeting on our side — there’s nothing else you need to do.\n"
            f"We had you down for: {when}.\n"
            f"{note_plain}\n"
            "If you’d ever like to find another time in the future, you’re always welcome to reach out through the site again.\n\n"
            "Warmly,\n"
            "Ada, Jayanth’s AI assistant"
        )
        html_r = self._email_shell(
            title="Meeting cancelled",
            subtitle="We’ve updated your request",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {html.escape(name)},</p>"
                "<p style='margin:0 0 12px;'>Thank you for letting us know. We’ve cancelled the meeting — "
                "there’s nothing else you need to do.</p>"
                f"<p style='margin:0 0 12px;'><strong>Previous time discussed:</strong> {html.escape(str(when))}</p>"
                + (note_html if note else "")
                + "<p style='margin:16px 0 0;'>If you’d like to connect another time in the future, you’re always welcome to reach out again.</p>"
            ),
        )
        sent_r = self._send_email(sub_r, body_r, req_email, html_r) if req_email else False

        sub_j = f"Meeting cancelled — {name}"
        body_j = (
            f"Hi Jayanth,\n\n"
            f"{name} has cancelled their meeting with you.\n"
            f"Time that had been discussed: {when}.\n"
            f"{note_plain}\n"
            "No further action needed unless you’d like to follow up personally.\n\n"
            "Warmly,\n"
            "Ada, Jayanth’s AI assistant"
        )
        html_j = self._email_shell(
            title="Meeting cancelled",
            subtitle=f"{name} withdrew the meeting",
            body_html=(
                f"<p style='margin:0 0 12px;'><strong>Requester:</strong> {html.escape(name)} "
                f"(<a href='mailto:{html.escape(req_email)}'>{html.escape(req_email)}</a>)</p>"
                f"<p style='margin:0 0 12px;'><strong>Previous time discussed:</strong> {html.escape(str(when))}</p>"
                + (note_html if note else "")
                + "<p style='margin:16px 0 0;'>No action required on your side unless you choose to follow up.</p>"
            ),
        )
        sent_j = False
        if notify_email:
            if req_email and req_email.lower() == notify_email.lower():
                sent_j = bool(sent_r)
            else:
                sent_j = self._send_email(sub_j, body_j, notify_email, html_j)
        return {"requester": sent_r, "host": sent_j}

    def _send_reschedule_emails(self, entry: Dict[str, Any]) -> Dict[str, bool]:
        """Let both sides know a new time is being arranged."""
        name = entry.get("name") or "there"
        req_email = (entry.get("email") or "").strip()
        notify_email = self._notify_email_default()
        base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip().rstrip("/")
        token = entry.get("token") or ""
        propose_url = f"{base_url}/api/meeting/propose/{token}"
        when = entry.get("proposed_time") or entry.get("preferred_time") or ""
        avail = (entry.get("reschedule_availability") or "").strip()
        note = (entry.get("reschedule_note") or "").strip()

        avail_plain = f"\nThey suggested: {avail}\n" if avail else ""
        note_plain = f"\nNote: {note}\n" if note else ""
        sub_r = "Reschedule request — Jayanth will propose a new time"
        body_r = (
            f"Hi {name},\n\n"
            "Thanks for your patience — I’ve let Jayanth know you’d like a different time. "
            "He’ll review your availability and email you a **formal proposed slot** (same process as when you first scheduled). "
            "The new time is **not** final until you confirm from that email.\n"
            f"{avail_plain}{note_plain}\n"
            "You don’t need to do anything else until you hear from him.\n\n"
            "Warmly,\n"
            "Ada, Jayanth’s AI assistant"
        )
        avail_html = (
            f"<p style='margin:0 0 12px;'><strong>Availability they shared:</strong> {html.escape(avail)}</p>"
            if avail
            else ""
        )
        note_html = (
            f"<p style='margin:0 0 12px;padding:12px;background:#f8fafc;border-radius:8px;'>{html.escape(note)}</p>"
            if note
            else ""
        )
        html_r = self._email_shell(
            title="Reschedule request received",
            subtitle="Jayanth will email a proposal to confirm",
            body_html=(
                f"<p style='margin:0 0 12px;'>Hi {html.escape(name)},</p>"
                "<p style='margin:0 0 12px;'>Thanks for letting us know — I’ve passed this to Jayanth. "
                "He’ll send a <strong>proposed time</strong> by email; please confirm there — same as your first booking.</p>"
                + (f"<p style='margin:0 0 8px;'><strong>Earlier discussion:</strong> {html.escape(str(when))}</p>" if when else "")
                + avail_html
                + note_html
                + "<p style='margin:16px 0 0;'>No action needed from you right now.</p>"
            ),
        )
        sent_r = self._send_email(sub_r, body_r, req_email, html_r) if req_email else False

        sub_j = f"Reschedule: propose a new time — {name}"
        body_j = (
            f"Hi Jayanth,\n\n"
            f"{name} asked to reschedule (preference below — not confirmed until you propose and they accept by email).\n"
            f"Previous time on record: {when or 'see thread'}.\n"
            f"{avail_plain}{note_plain}\n"
            f"Open your proposal form to send them a slot to confirm:\n{propose_url}\n\n"
            "Warmly,\n"
            "Ada, Jayanth’s AI assistant"
        )
        html_j = self._email_shell(
            title="Reschedule requested",
            subtitle=f"{name} asked for a different time",
            body_html=(
                f"<p style='margin:0 0 12px;'><strong>Requester:</strong> {html.escape(name)} — "
                f"<a href='mailto:{html.escape(req_email)}'>{html.escape(req_email)}</a></p>"
                + (f"<p style='margin:0 0 8px;'><strong>Earlier time:</strong> {html.escape(str(when))}</p>" if when else "")
                + avail_html
                + note_html
                + "<p style='margin:0 0 14px;'>Open your proposal form:</p>"
                + f"<p style='margin:0;'><a href='{html.escape(propose_url)}' "
                "style='display:inline-block;background:#1d4ed8;color:#fff;text-decoration:none;padding:10px 14px;border-radius:8px;'>"
                "Propose a new time</a></p>"
            ),
        )
        sent_j = False
        if notify_email:
            if req_email and req_email.lower() == notify_email.lower():
                sent_j = bool(sent_r)
            else:
                sent_j = self._send_email(sub_j, body_j, notify_email, html_j)
        return {"requester": sent_r, "host": sent_j}

    def cancel_meeting_request(self, request_id: str, note: Optional[str] = None) -> Dict[str, Any]:
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("request_id") != request_id:
                continue
            st = (entry.get("status") or "").strip()
            if st == "cancelled":
                return {"status": "already_cancelled", "request": dict(entry)}
            if st not in self._ACTIVE_FOR_CHANGE:
                return {"status": "invalid_state", "reason": f"Cannot cancel from status {st!r}"}
            entry["status"] = "cancelled"
            entry["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            entry["cancellation_note"] = (note or "").strip() or None
            self._save(payload)
            emails = self._send_cancellation_emails(entry)
            return {"status": "ok", "request": dict(entry), "emails": emails}
        return {"status": "not_found"}

    def request_reschedule_meeting(
        self,
        request_id: str,
        new_availability: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._load()
        for entry in payload.get("requests", []):
            if entry.get("request_id") != request_id:
                continue
            st = (entry.get("status") or "").strip()
            if st == "cancelled":
                return {"status": "invalid_state", "reason": "Meeting already cancelled"}
            if st not in self._ACTIVE_FOR_CHANGE:
                return {"status": "invalid_state", "reason": f"Cannot reschedule from status {st!r}"}
            # Same loop as initial booking: host must propose again; requester confirms by email — not a one-shot time change.
            entry["status"] = "awaiting_host_proposal"
            entry["reschedule_requested_at"] = datetime.now(timezone.utc).isoformat()
            entry["reschedule_note"] = (note or "").strip() or None
            entry["reschedule_availability"] = (new_availability or "").strip() or None
            self._save(payload)
            emails = self._send_reschedule_emails(entry)
            return {"status": "ok", "request": dict(entry), "emails": emails}
        return {"status": "not_found"}
